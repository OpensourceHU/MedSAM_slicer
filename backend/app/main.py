"""FastAPI entrypoint for MedSAM Slicer backend."""

import base64
from dataclasses import asdict
from functools import lru_cache
from io import BytesIO
import time

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.segmenter import SAM2Segmenter, compute_volume_bounds, slice_to_rgb_uint8
from app.volume import (
    Plane,
    extract_mask_mesh,
    extract_slice_png,
    extract_surface_mesh,
    extract_volume_bytes,
    list_volumes,
    load_volume_array,
)

app = FastAPI(title="MedSAM Slicer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/volumes")
def get_volumes() -> dict[str, list[str]]:
    return {"volumes": list_volumes()}


@app.get("/volumes/{filename}/info")
def get_volume_info(filename: str) -> dict:
    try:
        _, info = load_volume_array(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return asdict(info)


@app.get("/volumes/{filename}/data")
def get_volume_data(filename: str) -> Response:
    try:
        raw_bytes, wire_dtype = extract_volume_bytes(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(
        content=raw_bytes,
        media_type="application/octet-stream",
        headers={"X-Volume-Dtype": wire_dtype},
    )


@app.get("/volumes/{filename}/slice")
def get_volume_slice(
    filename: str,
    plane: Plane = Query("axial"),
    index: int = Query(0, ge=0),
    window: float | None = Query(None, gt=0),
    level: float | None = Query(None),
) -> Response:
    try:
        png_bytes = extract_slice_png(filename, plane, index, window=window, level=level)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(content=png_bytes, media_type="image/png")


@app.get("/volumes/{filename}/mesh")
def get_volume_mesh(
    filename: str,
    threshold: float | None = Query(None),
    step_size: int = Query(2, ge=1, le=6),
) -> dict:
    try:
        return extract_surface_mesh(filename, threshold=threshold, step_size=step_size)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@lru_cache(maxsize=1)
def get_segmenter() -> SAM2Segmenter:
    return SAM2Segmenter(variant="tiny")


class SegmentRequest(BaseModel):
    slice: int = Field(..., ge=0, description="Axial slice index z")
    points: list[list[float]] = Field(..., min_length=1, description="[[x, y], ...] in voxel coordinates")
    labels: list[int] = Field(..., min_length=1, description="1 = include, 0 = exclude")
    box: list[float] | None = Field(default=None, description="Optional [x0, y0, x1, y1] in voxel coordinates")


def _mask_to_png_base64(mask_2d: np.ndarray) -> str:
    """Encode a 2D boolean mask as a transparent cyan RGBA PNG base64 string."""
    h, w = mask_2d.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask_2d] = [56, 189, 248, 160]
    buf = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/volumes/{filename}/segment")
def segment_slice(filename: str, req: SegmentRequest) -> dict:
    if len(req.points) != len(req.labels):
        raise HTTPException(status_code=400, detail="points and labels must have the same length")

    try:
        vol, info = load_volume_array(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    nx, ny, nz = info.shape
    if not 0 <= req.slice < nz:
        raise HTTPException(status_code=400, detail=f"Axial slice {req.slice} out of range [0, {nz - 1}]")

    # Convert voxel [x, y] to np.rot90 slice pixel [x, ny - 1 - y]
    rot_points = [[p[0], ny - 1 - p[1]] for p in req.points]
    rot_box = None
    if req.box is not None and len(req.box) == 4:
        x0, y0, x1, y1 = req.box
        rot_box = [min(x0, x1), ny - 1 - max(y0, y1), max(x0, x1), ny - 1 - min(y0, y1)]

    low, high = compute_volume_bounds(vol)
    slc = np.rot90(vol[:, :, req.slice])
    rgb = slice_to_rgb_uint8(slc, low, high)

    result = get_segmenter().predict(
        volume_id=filename,
        slice_index=req.slice,
        rgb_image=rgb,
        points=rot_points,
        labels=req.labels,
        box=rot_box,
    )

    return {
        "slice": req.slice,
        "score": round(result.score, 4),
        "encode_ms": result.encode_ms,
        "predict_ms": result.predict_ms,
        "cache_hit": result.cache_hit,
        "mask_pixels": int(result.mask.sum()),
        "mask_png_base64": _mask_to_png_base64(result.mask),
    }


@app.post("/volumes/{filename}/propagate")
def propagate_volume(
    filename: str,
    req: SegmentRequest,
    max_steps: int = Query(15, ge=1, le=100),
) -> dict:
    if len(req.points) != len(req.labels):
        raise HTTPException(status_code=400, detail="points and labels must have the same length")

    try:
        vol, info = load_volume_array(filename)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    nx, ny, nz = info.shape
    if not 0 <= req.slice < nz:
        raise HTTPException(status_code=400, detail=f"Axial slice {req.slice} out of range [0, {nz - 1}]")

    rot_points = [[p[0], ny - 1 - p[1]] for p in req.points]

    t0 = time.perf_counter()
    masks_by_slice = get_segmenter().propagate(
        volume_id=filename,
        vol_ras=vol,
        start_slice=req.slice,
        points=rot_points,
        labels=req.labels,
        max_steps=max_steps,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Assemble 3D RAS+ mask volume for statistics & 3D surface preview (M10, M11)
    mask_3d = np.zeros((nx, ny, nz), dtype=np.float32)
    for z, mask_2d in masks_by_slice.items():
        mask_3d[:, :, z] = np.rot90(mask_2d, k=-1).astype(np.float32)

    voxel_count = int(mask_3d.sum())
    sx, sy, sz = info.spacing
    volume_ml = round(voxel_count * sx * sy * sz / 1000.0, 3)
    mask_mesh = extract_mask_mesh(mask_3d, info.spacing, step_size=1)

    sorted_z = sorted(masks_by_slice.keys())
    encoded_slices = {
        str(z): _mask_to_png_base64(masks_by_slice[z])
        for z in sorted_z
    }
    return {
        "start_slice": req.slice,
        "slice_range": [sorted_z[0], sorted_z[-1]] if sorted_z else [req.slice, req.slice],
        "slice_count": len(encoded_slices),
        "voxel_count": voxel_count,
        "volume_ml": volume_ml,
        "elapsed_ms": elapsed_ms,
        "slices": encoded_slices,
        "mask_mesh": mask_mesh,
    }
