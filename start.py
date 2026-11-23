"""Cross-platform dev launcher for MedSAM Slicer (macOS, Linux/WSL2, Windows).

Starts both the FastAPI backend (port 8000) and the frontend static server (port 3000),
and terminates both cleanly on Ctrl+C.

Usage:
    python3 start.py
    # or: uv run --project backend python start.py
"""

from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = REPO_ROOT / "backend"
FRONTEND_DIR = REPO_ROOT / "frontend"


def find_uv() -> str:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in (
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
    ):
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "Could not find 'uv'. Install it from https://docs.astral.sh/uv/ or add it to PATH."
    )


def main() -> int:
    uv_bin = find_uv()

    if not (BACKEND_DIR / ".venv").exists():
        print("[setup] Creating backend/.venv via uv sync...")
        subprocess.run([uv_bin, "sync", "--directory", str(BACKEND_DIR)], check=True)

    backend_cmd = [uv_bin, "run", "fastapi", "dev", "app/main.py", "--port", "8000"]
    frontend_cmd = [
        uv_bin,
        "run",
        "--project",
        str(BACKEND_DIR),
        "python",
        "-m",
        "http.server",
        "3000",
        "--directory",
        str(FRONTEND_DIR),
    ]

    print("Starting MedSAM Slicer...")
    print("  Backend API : http://127.0.0.1:8000/docs")
    print("  Frontend UI : http://127.0.0.1:3000")
    print("Press Ctrl+C to stop both servers.\n")

    backend_proc = subprocess.Popen(backend_cmd, cwd=BACKEND_DIR)
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=REPO_ROOT)
    procs = [backend_proc, frontend_proc]

    def shutdown(*_args) -> None:
        print("\nStopping servers...")
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    shutdown()
                    return code
            time.sleep(0.5)
    except KeyboardInterrupt:
        shutdown()
        return 0


if __name__ == "__main__":
    sys.exit(main())
