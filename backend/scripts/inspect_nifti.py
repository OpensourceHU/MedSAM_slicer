"""Quick script to inspect and slice .nii.gz volumes in Volumes_dataset_export."""

from pathlib import Path
import nibabel as nib
import numpy as np


DATA_DIR = Path(__file__).resolve().parents[2] / "Volumes_dataset_export"


def inspect_all_volumes(data_dir: Path) -> None:
    files = sorted(data_dir.glob("*.nii.gz"))
    if not files:
        print(f"No .nii.gz files found in {data_dir}")
        return

    for path in files:
        img = nib.load(path)
        data = img.get_fdata()
        spacing = tuple(round(float(z), 3) for z in img.header.get_zooms())
        print(
            f"{path.name:<20} | shape={str(data.shape):<18} "
            f"| spacing={str(spacing):<22} "
            f"| raw_dtype={str(img.get_data_dtype()):<8} "
            f"| min={data.min():.1f}, max={data.max():.1f}"
        )


def apply_window_level(slice_2d: np.ndarray, window: float, level: float) -> np.ndarray:
    """Map raw MRI/CT intensities to [0, 255] uint8 using Window/Level."""
    low = level - window / 2.0
    high = level + window / 2.0
    clipped = np.clip(slice_2d, low, high)
    norm = (clipped - low) / max(high - low, 1e-6)
    return (norm * 255.0).astype(np.uint8)


def preview_three_planes(nifti_path: Path) -> None:
    """Extract middle Axial, Coronal, and Sagittal slices and save a PNG preview."""
    import matplotlib.pyplot as plt

    img = nib.as_closest_canonical(nib.load(nifti_path))
    vol = img.get_fdata()

    x, y, z = vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2

    sagittal = np.rot90(vol[x, :, :])
    coronal = np.rot90(vol[:, y, :])
    axial = np.rot90(vol[:, :, z])

    p1, p99 = np.percentile(vol, (1, 99))
    window, level = p99 - p1, (p99 + p1) / 2.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, slc, title in zip(
        axes,
        [axial, coronal, sagittal],
        [f"Axial (z={z})", f"Coronal (y={y})", f"Sagittal (x={x})"],
    ):
        ax.imshow(apply_window_level(slc, window, level), cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(nifti_path.name)
    out_path = Path(__file__).resolve().parent / f"{nifti_path.name.replace('.nii.gz', '')}_preview.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved 3-plane preview to: {out_path}")


if __name__ == "__main__":
    inspect_all_volumes(DATA_DIR)
    sample = DATA_DIR / "chris_t1.nii.gz"
    if sample.exists():
        preview_three_planes(sample)
