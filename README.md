# MedSAM Slicer

Interactive 3D medical image viewer and prompt-driven segmentation tool built for the Visual Computing course project.

---

## Project Constraints & Conventions

1. **English-Only Codebase & UI**
   - All backend code, frontend code, comments, docstrings, API schemas, and user-facing UI text **must be written in English**.
2. **Python Backend & `uv` Package Management**
   - The backend is written in **Python** (FastAPI + `nibabel` + `numpy` + `scikit-image` + `pillow`).
   - All Python dependencies, virtual environments, and script executions **must be managed with `uv`** via `backend/pyproject.toml` and `backend/.python-version` (Python 3.12).
   - Do not use `pip` directly to modify the environment; use `uv add <package>` / `uv sync`.

---

## Project Structure

```text
VC_visual_computing/
├── README.md
├── MedSAM_Slicer_Project-2 (2).pdf
├── Volumes_dataset_export/          # Sample .nii.gz datasets
├── backend/
│   ├── .python-version              # Pinned Python 3.12 for uv
│   ├── pyproject.toml               # uv dependency manifest
│   ├── uv.lock                      # Locked dependency tree
│   ├── app/
│   │   ├── main.py                  # Pure FastAPI backend server
│   │   └── volume.py                # NIfTI I/O, 3-plane slicing, W/L, Marching Cubes
│   └── scripts/
│       └── inspect_nifti.py         # CLI script to inspect dataset headers & preview
└── frontend/                        # Standalone HTML5 Canvas + Three.js web app
    ├── index.html
    ├── style.css
    └── app.js
```

---

## Setup & Running Locally

### 1. Install backend dependencies with `uv`

If `uv` is installed at `~/.local/bin/uv` (default on macOS):

```bash
cd backend
~/.local/bin/uv sync
```

*(If `uv` is already in your `PATH`, you can simply run `uv sync`.)*

### 2. Start the backend API server (Terminal 1)

```bash
cd backend
~/.local/bin/uv run fastapi dev app/main.py
```

- **Swagger API Documentation**: http://127.0.0.1:8000/docs

### 3. Start the frontend static server (Terminal 2)

```bash
python3 -m http.server 3000 --directory frontend
```

- **Interactive 3-Plane + 3D Viewer**: http://127.0.0.1:3000
- On Windows, if `python3` is not found: `uv run --project backend python -m http.server 3000 --directory frontend` (from the repo root)

### 3b. Test volumes for the WebGL axial view (until `/data` exists)

The axial view is drawn in the browser with WebGL from the whole volume. Until the backend has a
`/volumes/{file}/data` endpoint, write the volumes once as `frontend/dev/<name>.bin` (float32, x fastest;
the folder is ignored by Git). From the repo root:

```bash
uv run --project backend python frontend/tools/make_dev_volume.py
```

A volume without a `.bin` file (and without `/data`) is still shown with the PNG slices. The Volume Info
panel says which one is used ("Axial view: WebGL (dev file)" or "PNG slices").

### 4. Inspect NIfTI datasets via CLI (optional)

```bash
cd backend
~/.local/bin/uv run python scripts/inspect_nifti.py
```

---

## Current Features (Phase 1)

- **NIfTI Loader**: Reads `.nii.gz` volumes with `nibabel` and normalizes orientation to canonical `RAS+`.
- **Linked 3-Plane Viewer**: Coronal, Sagittal, and Axial views rendered on HTML5 Canvas with synchronized crosshairs, click/drag navigation, and mouse-wheel scrolling.
- **Window / Level Controls**: Real-time intensity windowing for MRI and 16-bit CT volumes.
- **3D Surface Reconstruction**: Real-time iso-surface mesh extraction using `skimage.measure.marching_cubes`, rendered with Three.js (`OrbitControls`) and synchronized 3D crosshairs.
- **WebGL Axial View** (week 1): the axial slice is drawn by a WebGL2 shader from the whole volume in GPU memory (3D texture), so scrolling and window/level need no server request; it fills its panel at the true voxel proportions. Falls back to PNG slices when no volume data is available.
- **Click Points** (week 1): Navigate / Include / Exclude tools (keys N, I, E). Include and exclude points on the axial view are stored in voxel coordinates and shown as green / red dots on their own slice; right-click adds the opposite point, Ctrl+Z undoes. `medsam.promptsForSlice()` in the browser console gives `{slice, points, labels}`, the format planned for the SAM2 prompts.

---

## Key API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check |
| `GET` | `/volumes` | List available `.nii.gz` files |
| `GET` | `/volumes/{filename}/info` | Volume shape, voxel spacing, intensity range, default W/L |
| `GET` | `/volumes/{filename}/slice` | Render a 2D slice PNG (`plane`, `index`, `window`, `level`) |
| `GET` | `/volumes/{filename}/mesh` | Extract 3D Marching Cubes surface mesh (`threshold`, `step_size`) |
# MedSAM_slicer
