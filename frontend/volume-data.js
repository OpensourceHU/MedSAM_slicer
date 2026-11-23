// Gets the whole volume into the browser once, as a Float32Array (x fastest).
//
// 1. Kane's /volumes/{file}/data endpoint, once it exists. Its dtype comes from
//    /info ("dtype"); int16 is converted to float32 here, so the shader only
//    ever sees float32.
// 2. Until then: frontend/dev/<name>.bin, written by tools/make_dev_volume.py.
// 3. Neither found: returns null, and the app keeps using the PNG slices.

import { API_BASE } from "./config.js";

const ARRAY_TYPES = { int16: Int16Array, uint16: Uint16Array, float32: Float32Array };

export function devName(filename) {
  return filename.replace(/\.nii(\.gz)?$/, "");
}

async function tryFetch(url) {
  try {
    const res = await fetch(url);
    return res.ok ? res : null;
  } catch {
    return null;
  }
}

function toFloat32(buffer, dtype) {
  const ArrayType = ARRAY_TYPES[dtype];
  if (!ArrayType) throw new Error(`Unsupported volume dtype: ${dtype}`);
  const raw = new ArrayType(buffer);
  return raw instanceof Float32Array ? raw : Float32Array.from(raw);
}

export async function loadVolumeData(filename, info) {
  const [nx, ny, nz] = info.shape;
  const expected = nx * ny * nz;

  const fromServer = await tryFetch(`${API_BASE}/volumes/${encodeURIComponent(filename)}/data`);
  if (fromServer) {
    const data = toFloat32(await fromServer.arrayBuffer(), info.dtype ?? "int16");
    if (data.length === expected) return { data, source: "server" };
    console.warn(`/data returned ${data.length} voxels, expected ${expected}`);
  }

  const fromDev = await tryFetch(`dev/${devName(filename)}.bin`);
  if (fromDev) {
    const data = toFloat32(await fromDev.arrayBuffer(), "float32");
    if (data.length === expected) return { data, source: "dev file" };
    console.warn(`dev/${devName(filename)}.bin has ${data.length} voxels, expected ${expected}`);
  }

  return null;
}
