"""FastAPI entrypoint for MedSAM Slicer backend."""

from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from app.volume import (
    Plane,
    extract_slice_png,
    extract_surface_mesh,
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
