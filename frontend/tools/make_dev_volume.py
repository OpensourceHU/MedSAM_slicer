"""Write test volumes for the WebGL viewer until the backend has a /data endpoint.

For each NIfTI file in Volumes_dataset_export/ this writes frontend/dev/<name>.bin:
the whole volume as float32, x fastest (Fortran order), in RAS orientation, exactly
what the browser uploads as a 3D texture. float32 because some scans (e.g. the Ernie
T1, values up to 65535 with decimals) do not fit int16.

Run from the repo root:
    uv run --project backend python frontend/tools/make_dev_volume.py              # all volumes
    uv run --project backend python frontend/tools/make_dev_volume.py ernie_T1.nii.gz
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_DIR = REPO / "Volumes_dataset_export"
OUT_DIR = REPO / "frontend" / "dev"


def stem(name: str) -> str:
    """'ernie_T1.nii.gz' -> 'ernie_T1' (the front end uses the same rule)."""
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def convert(path: Path) -> None:
    img = nib.as_closest_canonical(nib.load(path))  # same orientation as the backend
    vol = img.get_fdata(dtype=np.float32)
    out = OUT_DIR / f"{stem(path.name)}.bin"
    out.write_bytes(vol.tobytes(order="F"))  # x fastest
    print(
        f"{path.name}: shape {vol.shape}, spacing {tuple(float(z) for z in img.header.get_zooms()[:3])}, "
        f"range {vol.min():.1f} to {vol.max():.1f} -> {out.relative_to(REPO)} "
        f"({out.stat().st_size:,} bytes)"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:]
    files = (
        [DATA_DIR / n for n in names]
        if names
        else sorted(
            p for p in DATA_DIR.iterdir() if p.name.endswith((".nii", ".nii.gz"))
        )
    )
    if not files:
        print(f"No .nii or .nii.gz files found in {DATA_DIR}")
    for path in files:
        convert(path)


if __name__ == "__main__":
    main()
