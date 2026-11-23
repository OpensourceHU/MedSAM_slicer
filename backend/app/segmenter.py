"""Decoupled SAM 2.1 segmentation interface (encode, predict, propagate)."""

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time
from typing import Sequence

import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"

MODEL_CONFIGS = {
    "tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam2.1_hiera_tiny.pt"),
    "small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam2.1_hiera_small.pt"),
}


@dataclass(frozen=True)
class PredictionResult:
    mask: np.ndarray  # 2D bool array matching slice (H, W)
    score: float  # SAM2 predicted IoU
    encode_ms: float
    predict_ms: float
    cache_hit: bool


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def compute_volume_bounds(vol: np.ndarray) -> tuple[float, float]:
    """Fixed per-volume 0.5-99.5 percentile bounds so W/L changes don't invalidate embeddings."""
    low, high = np.percentile(vol, (0.5, 99.5))
    return float(low), float(max(high, low + 1e-6))


def slice_to_rgb_uint8(slice_2d: np.ndarray, low: float, high: float) -> np.ndarray:
    """Normalize a 2D slice using fixed volume bounds and expand to 3-channel uint8 RGB."""
    clipped = np.clip(slice_2d, low, high)
    norm = ((clipped - low) / (high - low) * 255.0).astype(np.uint8)
    return np.stack([norm, norm, norm], axis=-1)


class SAM2Segmenter:
    """Wrapper around SAM 2.1 image predictor with per-slice embedding cache."""

    def __init__(self, variant: str = "tiny", device: str | None = None) -> None:
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported SAM2 variant: {variant}")
        cfg_name, ckpt_name = MODEL_CONFIGS[variant]
        ckpt_path = CHECKPOINTS_DIR / ckpt_name
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. Run scripts/download_checkpoints.py first."
            )

        self.variant = variant
        self.device = device or select_device()
        model = build_sam2(cfg_name, str(ckpt_path), device=self.device)
        self.predictor = SAM2ImagePredictor(model)
        self._video_predictor = None
        self._cfg_name = cfg_name
        self._ckpt_path = str(ckpt_path)
        self._cache: dict[tuple[str, int], tuple[dict, tuple[int, int]]] = {}
        self._frame_dirs: dict[str, Path] = {}
        self._video_states: dict[str, dict] = {}

    def clear_cache(self, volume_id: str | None = None) -> None:
        if volume_id is None:
            self._cache.clear()
            self._video_states.clear()
        else:
            keys = [k for k in self._cache if k[0] == volume_id]
            for k in keys:
                del self._cache[k]
            self._video_states.pop(volume_id, None)

    def encode(self, volume_id: str, slice_index: int, rgb_image: np.ndarray) -> float:
        """Encode a single RGB slice and cache its SAM2 image features. Returns encode_ms."""
        key = (volume_id, slice_index)
        if key in self._cache:
            return 0.0

        t0 = time.perf_counter()
        with torch.inference_mode():
            self.predictor.set_image(rgb_image)
            features = {
                k: v.detach() if isinstance(v, torch.Tensor) else v
                for k, v in self.predictor._features.items()
            }
            orig_hw = self.predictor._orig_hw
        encode_ms = (time.perf_counter() - t0) * 1000.0
        self._cache[key] = (features, orig_hw)
        return encode_ms

    def predict(
        self,
        volume_id: str,
        slice_index: int,
        rgb_image: np.ndarray,
        points: Sequence[Sequence[float]],
        labels: Sequence[int],
        box: Sequence[float] | None = None,
    ) -> PredictionResult:
        """Run SAM2 mask decoder on a slice using cached embeddings when available."""
        key = (volume_id, slice_index)
        cache_hit = key in self._cache
        encode_ms = 0.0 if cache_hit else self.encode(volume_id, slice_index, rgb_image)

        features, orig_hw = self._cache[key]
        self.predictor._features = features
        self.predictor._orig_hw = orig_hw
        self.predictor._is_image_set = True

        pts_arr = np.asarray(points, dtype=np.float32) if points else None
        lbl_arr = np.asarray(labels, dtype=np.int32) if labels else None
        box_arr = np.asarray(box, dtype=np.float32) if box is not None else None

        t0 = time.perf_counter()
        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords=pts_arr,
                point_labels=lbl_arr,
                box=box_arr,
                multimask_output=True,
            )
        predict_ms = (time.perf_counter() - t0) * 1000.0

        best_idx = int(np.argmax(scores))
        return PredictionResult(
            mask=masks[best_idx].astype(bool),
            score=float(scores[best_idx]),
            encode_ms=round(encode_ms, 2),
            predict_ms=round(predict_ms, 2),
            cache_hit=cache_hit,
        )

    def _get_video_predictor(self):
        if self._video_predictor is None:
            self._video_predictor = build_sam2_video_predictor(
                self._cfg_name, self._ckpt_path, device=self.device
            )
        return self._video_predictor

    def _ensure_frame_dir(self, volume_id: str, vol_ras: np.ndarray) -> Path:
        """Export axial slices as JPEGs once per volume for SAM2VideoPredictor."""
        if volume_id in self._frame_dirs and self._frame_dirs[volume_id].exists():
            return self._frame_dirs[volume_id]

        frame_dir = Path(tempfile.mkdtemp(prefix="medsam_frames_"))
        low, high = compute_volume_bounds(vol_ras)
        nz = vol_ras.shape[2]
        for z in range(nz):
            slc = np.rot90(vol_ras[:, :, z])
            rgb = slice_to_rgb_uint8(slc, low, high)
            Image.fromarray(rgb, mode="RGB").save(frame_dir / f"{z:05d}.jpg", quality=95)
        self._frame_dirs[volume_id] = frame_dir
        return frame_dir

    def _get_video_state(self, volume_id: str, vol_ras: np.ndarray) -> dict:
        """Cache SAM2VideoPredictor inference_state per volume to avoid reloading JPEGs."""
        video_pred = self._get_video_predictor()
        if volume_id not in self._video_states:
            frame_dir = self._ensure_frame_dir(volume_id, vol_ras)
            self._video_states[volume_id] = video_pred.init_state(
                video_path=str(frame_dir)
            )
        else:
            video_pred.reset_state(self._video_states[volume_id])
        return self._video_states[volume_id]

    def propagate(
        self,
        volume_id: str,
        vol_ras: np.ndarray,
        start_slice: int,
        points: Sequence[Sequence[float]],
        labels: Sequence[int],
        max_steps: int = 20,
        min_pixels: int = 10,
    ) -> dict[int, np.ndarray]:
        """Bidirectional slice propagation from start_slice using SAM2VideoPredictor."""
        video_pred = self._get_video_predictor()
        pts_arr = np.asarray(points, dtype=np.float32)
        lbl_arr = np.asarray(labels, dtype=np.int32)
        masks_by_slice: dict[int, np.ndarray] = {}

        amp_ctx = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else torch.inference_mode()
        )
        with torch.inference_mode(), amp_ctx:
            state = self._get_video_state(volume_id, vol_ras)
            _, out_obj_ids, out_mask_logits = video_pred.add_new_points_or_box(
                inference_state=state,
                frame_idx=start_slice,
                obj_id=1,
                points=pts_arr,
                labels=lbl_arr,
            )
            masks_by_slice[start_slice] = (out_mask_logits[0, 0] > 0.0).cpu().numpy()

            for reverse in (False, True):
                for frame_idx, _obj_ids, mask_logits in video_pred.propagate_in_video(
                    state,
                    start_frame_idx=start_slice,
                    max_frame_num_to_track=max_steps,
                    reverse=reverse,
                ):
                    mask_2d = (mask_logits[0, 0] > 0.0).cpu().numpy()
                    if frame_idx != start_slice and int(mask_2d.sum()) < min_pixels:
                        break
                    masks_by_slice[int(frame_idx)] = mask_2d

        return masks_by_slice
