"""NIfTI volume loading, caching, 3-plane slicing, and Window/Level rendering."""

from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from PIL import Image
from skimage.measure import marching_cubes

Plane = Literal["axial", "coronal", "sagittal"]

DATA_DIR = Path(__file__).resolve().parents[2] / "Volumes_dataset_export"


@dataclass(frozen=True)
class VolumeInfo:
    filename: str
    shape: tuple[int, int, int]
    spacing: tuple[float, float, float]
    min_val: float
    max_val: float
    default_window: float
    default_level: float


def list_volumes(data_dir: Path = DATA_DIR) -> list[str]:
    if not data_dir.exists():
        return []
    return sorted(p.name for p in data_dir.glob("*.nii.gz"))


def resolve_volume_path(filename: str, data_dir: Path = DATA_DIR) -> Path:
    safe_name = Path(filename).name
    path = data_dir / safe_name
    if not path.exists() or not safe_name.endswith(".nii.gz"):
        raise FileNotFoundError(f"Volume not found: {safe_name}")
    return path


@lru_cache(maxsize=4)
def load_volume_array(filename: str) -> tuple[np.ndarray, VolumeInfo]:
    path = resolve_volume_path(filename)
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.get_fdata(), dtype=np.float32)

    zooms = img.header.get_zooms()[:3]
    spacing = tuple(round(float(z), 4) for z in zooms)
    p1, p99 = np.percentile(vol, (1, 99))
    default_window = float(max(p99 - p1, 1.0))
    default_level = float((p99 + p1) / 2.0)

    info = VolumeInfo(
        filename=path.name,
        shape=(int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
        spacing=(spacing[0], spacing[1], spacing[2]),
        min_val=float(vol.min()),
        max_val=float(vol.max()),
        default_window=round(default_window, 2),
        default_level=round(default_level, 2),
    )
    return vol, info


def apply_window_level(slice_2d: np.ndarray, window: float, level: float) -> np.ndarray:
    low = level - window / 2.0
    high = level + window / 2.0
    clipped = np.clip(slice_2d, low, high)
    norm = (clipped - low) / max(high - low, 1e-6)
    return (norm * 255.0).astype(np.uint8)


def extract_slice_png(
    filename: str,
    plane: Plane,
    index: int,
    window: float | None = None,
    level: float | None = None,
) -> bytes:
    vol, info = load_volume_array(filename)
    nx, ny, nz = info.shape

    if plane == "axial":
        if not 0 <= index < nz:
            raise ValueError(f"Axial index {index} out of range [0, {nz - 1}]")
        slc = np.rot90(vol[:, :, index])
    elif plane == "coronal":
        if not 0 <= index < ny:
            raise ValueError(f"Coronal index {index} out of range [0, {ny - 1}]")
        slc = np.rot90(vol[:, index, :])
    elif plane == "sagittal":
        if not 0 <= index < nx:
            raise ValueError(f"Sagittal index {index} out of range [0, {nx - 1}]")
        slc = np.rot90(vol[index, :, :])
    else:
        raise ValueError(f"Unsupported plane: {plane}")

    w = window if window is not None and window > 0 else info.default_window
    l = level if level is not None else info.default_level
    img_u8 = apply_window_level(slc, w, l)

    buf = BytesIO()
    Image.fromarray(img_u8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def extract_surface_mesh(
    filename: str,
    threshold: float | None = None,
    step_size: int = 2,
) -> dict:
    """Run Marching Cubes on the volume to produce a centered 3D triangle mesh."""
    vol, info = load_volume_array(filename)

    if threshold is None:
        # Pick a sensible default above background (35% between p5 and p99)
        p5, p99 = np.percentile(vol, (5, 99))
        level = float(p5 + 0.35 * (p99 - p5))
    else:
        level = float(threshold)

    if level <= info.min_val or level >= info.max_val:
        return {
            "threshold": round(level, 2),
            "vertex_count": 0,
            "face_count": 0,
            "vertices": [],
            "faces": [],
        }

    verts, faces, _, _ = marching_cubes(
        vol,
        level=level,
        spacing=info.spacing,
        step_size=max(1, step_size),
    )

    # Center mesh around origin so Three.js OrbitControls rotates around volume center
    center = np.array(
        [
            (info.shape[0] - 1) * info.spacing[0] * 0.5,
            (info.shape[1] - 1) * info.spacing[1] * 0.5,
            (info.shape[2] - 1) * info.spacing[2] * 0.5,
        ],
        dtype=np.float32,
    )
    verts = verts - center

    return {
        "threshold": round(level, 2),
        "vertex_count": int(len(verts)),
        "face_count": int(len(faces)),
        "vertices": np.round(verts.reshape(-1), 2).tolist(),
        "faces": faces.astype(np.int32).reshape(-1).tolist(),
    }
