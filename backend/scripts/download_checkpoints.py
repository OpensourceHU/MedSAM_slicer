"""Download official Meta SAM 2.1 model checkpoints into backend/checkpoints/.

Usage:
    uv run --project backend python backend/scripts/download_checkpoints.py
    uv run --project backend python backend/scripts/download_checkpoints.py --model small
"""

import argparse
import urllib.request
from pathlib import Path

CHECKPOINTS_DIR = Path(__file__).resolve().parents[1] / "checkpoints"

URLS = {
    "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
    "small": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
}


def download_checkpoint(model: str) -> Path:
    url = URLS[model]
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = CHECKPOINTS_DIR / Path(url).name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"Checkpoint already exists: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    print(f"Downloading {model} checkpoint from {url} ...")
    tmp = dest.with_suffix(".pt.part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(dest)
    print(f"Saved to: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SAM 2.1 checkpoints.")
    parser.add_argument(
        "--model",
        choices=sorted(URLS.keys()),
        default="tiny",
        help="Which SAM 2.1 checkpoint to download (default: tiny).",
    )
    args = parser.parse_args()
    download_checkpoint(args.model)


if __name__ == "__main__":
    main()
