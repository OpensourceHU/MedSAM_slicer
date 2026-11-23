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
    dtype: Literal["int16", "float32"]
    affine: list[list[float]]


def list_volumes(data_dir: Path | None = None) -> list[str]:
    base = data_dir or DATA_DIR
    if not base.exists():
        return []
    return sorted(
        p.name for p in base.iterdir() if p.name.endswith((".nii", ".nii.gz"))
    )


def resolve_volume_path(filename: str, data_dir: Path | None = None) -> Path:
    base = data_dir or DATA_DIR
    safe_name = Path(filename).name
    path = base / safe_name
    if not path.exists() or not safe_name.endswith((".nii", ".nii.gz")):
        raise FileNotFoundError(f"Volume not found: {safe_name}")
    return path


def _select_wire_dtype(vol: np.ndarray, min_val: float, max_val: float) -> Literal["int16", "float32"]:
    """Use int16 when all voxels are whole numbers within [-32768, 32767], else float32."""
    if (
        min_val >= -32768.0
        and max_val <= 32767.0
        and bool(np.all(np.isfinite(vol)))
        and bool(np.array_equal(vol, np.trunc(vol)))
    ):
        return "int16"
    return "float32"


@lru_cache(maxsize=4)
def load_volume_array(filename: str) -> tuple[np.ndarray, VolumeInfo]:
    path = resolve_volume_path(filename)
    img = nib.as_closest_canonical(nib.load(path))
    vol = np.asarray(img.get_fdata(), dtype=np.float32)

    zooms = img.header.get_zooms()[:3]
    spacing = tuple(round(float(z), 4) for z in zooms)
    min_val = float(vol.min())
    max_val = float(vol.max())
    p1, p99 = np.percentile(vol, (1, 99))
    default_window = float(max(p99 - p1, 1.0))
    default_level = float((p99 + p1) / 2.0)
    wire_dtype = _select_wire_dtype(vol, min_val, max_val)
    affine = np.round(img.affine, 6).tolist()

    info = VolumeInfo(
        filename=path.name,
        shape=(int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])),
        spacing=(spacing[0], spacing[1], spacing[2]),
        min_val=min_val,
        max_val=max_val,
        default_window=round(default_window, 2),
        default_level=round(default_level, 2),
        dtype=wire_dtype,
        affine=affine,
    )
    return vol, info


def extract_volume_bytes(filename: str) -> tuple[bytes, Literal["int16", "float32"]]:
    """Serialize the canonical RAS+ volume in Fortran order (x fastest) for WebGL."""
    vol, info = load_volume_array(filename)
    target_dtype = np.int16 if info.dtype == "int16" else np.float32
    return vol.astype(target_dtype).tobytes(order="F"), info.dtype


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


def _run_marching_cubes(
    vol_3d: np.ndarray,
    level: float,
    spacing: tuple[float, float, float],
    step_size: int = 2,
) -> dict:
    """Run Marching Cubes and center the resulting mesh around the volume origin."""
    verts, faces, _, _ = marching_cubes(
        vol_3d,
        level=level,
        spacing=spacing,
        step_size=max(1, step_size),
    )
    center = np.array(
        [
            (vol_3d.shape[0] - 1) * spacing[0] * 0.5,
            (vol_3d.shape[1] - 1) * spacing[1] * 0.5,
            (vol_3d.shape[2] - 1) * spacing[2] * 0.5,
        ],
        dtype=np.float32,
    )
    verts = verts - center
    return {
        "threshold": round(float(level), 2),
        "vertex_count": int(len(verts)),
        "face_count": int(len(faces)),
        "vertices": np.round(verts.reshape(-1), 2).tolist(),
        "faces": faces.astype(np.int32).reshape(-1).tolist(),
    }


def extract_surface_mesh(
    filename: str,
    threshold: float | None = None,
    step_size: int = 2,
) -> dict:
    """Run Marching Cubes on the volume to produce a centered 3D triangle mesh."""
    vol, info = load_volume_array(filename)

    if threshold is None:
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

    return _run_marching_cubes(vol, level, info.spacing, step_size=step_size)


def extract_mask_mesh(
    mask_3d: np.ndarray,
    spacing: tuple[float, float, float],
    step_size: int = 1,
) -> dict:
    """Extract a 3D surface mesh from a binary segmentation mask volume."""
    if mask_3d.max() <= 0.5 or mask_3d.min() >= 0.5:
        return {"threshold": 0.5, "vertex_count": 0, "face_count": 0, "vertices": [], "faces": []}
    try:
        return _run_marching_cubes(mask_3d, 0.5, spacing, step_size=step_size)
    except (ValueError, RuntimeError):
        return {"threshold": 0.5, "vertex_count": 0, "face_count": 0, "vertices": [], "faces": []}
