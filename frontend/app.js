import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { API_BASE } from "./config.js";
import { SliceView } from "./slice-view.js";
import { loadVolumeData } from "./volume-data.js";

const state = {
  filename: null,
  info: null,
  coords: { x: 0, y: 0, z: 0 },
  window: 255,
  level: 127,
  threshold: 60,
  wireframe: false,
  images: { axial: null, coronal: null, sagittal: null },
  useGL: false, // axial view drawn by WebGL from the volume (true) or from PNG slices (false)
  tool: "navigate", // "navigate" | "include" | "exclude"
  prompts: [], // click points: { x, y, z, label } with label 1 = include, 0 = exclude
  masks: {}, // { [z]: HTMLImageElement } from /segment
};

const els = {
  volumeSelect: document.getElementById("volumeSelect"),
  coordReadout: document.getElementById("coordReadout"),
  metaShape: document.getElementById("metaShape"),
  metaSpacing: document.getElementById("metaSpacing"),
  metaRange: document.getElementById("metaRange"),
  windowSlider: document.getElementById("windowSlider"),
  levelSlider: document.getElementById("levelSlider"),
  windowVal: document.getElementById("windowVal"),
  levelVal: document.getElementById("levelVal"),
  resetWLBtn: document.getElementById("resetWLBtn"),
  axialSlider: document.getElementById("axialSlider"),
  coronalSlider: document.getElementById("coronalSlider"),
  sagittalSlider: document.getElementById("sagittalSlider"),
  axialVal: document.getElementById("axialVal"),
  coronalVal: document.getElementById("coronalVal"),
  sagittalVal: document.getElementById("sagittalVal"),
  axialLabel: document.getElementById("axialLabel"),
  coronalLabel: document.getElementById("coronalLabel"),
  sagittalLabel: document.getElementById("sagittalLabel"),
  thresholdSlider: document.getElementById("thresholdSlider"),
  thresholdVal: document.getElementById("thresholdVal"),
  wireframeToggle: document.getElementById("wireframeToggle"),
  meshStats: document.getElementById("meshStats"),
  meshStatusLabel: document.getElementById("meshStatusLabel"),
  threeContainer: document.getElementById("threeContainer"),
  canvases: {
    axial: document.getElementById("axialCanvas"),
    coronal: document.getElementById("coronalCanvas"),
    sagittal: document.getElementById("sagittalCanvas"),
  },
  axialGL: document.getElementById("axialGL"),
  axialStack: document.getElementById("axialStack"),
  metaRender: document.getElementById("metaRender"),
  toolButtons: document.querySelectorAll(".tool-btn"),
  activeToolLabel: document.getElementById("activeToolLabel"),
  promptInfo: document.getElementById("promptInfo"),
  undoPromptBtn: document.getElementById("undoPromptBtn"),
  clearPromptsBtn: document.getElementById("clearPromptsBtn"),
  propagateBtn: document.getElementById("propagateBtn"),
};

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

// --- WebGL axial view (falls back to Kane's PNG slices if WebGL2 is missing) ---
let axialView = null;
try {
  axialView = new SliceView(els.axialGL, els.canvases.axial, "axial");
} catch (err) {
  console.warn("Axial view falls back to PNG slices:", err);
}

function setRenderMode(useGL, source) {
  state.useGL = useGL;
  els.axialStack.classList.toggle("png-mode", !useGL);
  if (!useGL) {
    axialView?.deactivate();
    els.axialStack.style.width = "";
    els.axialStack.style.height = "";
  }
  els.metaRender.textContent = useGL ? `WebGL (${source})` : "PNG slices";
}

// --- Click points (prompts for SAM2 later) ---
function setTool(tool) {
  state.tool = tool;
  for (const btn of els.toolButtons) btn.classList.toggle("active", btn.dataset.tool === tool);
  els.activeToolLabel.textContent = tool[0].toUpperCase() + tool.slice(1);
  els.canvases.axial.dataset.tool = tool;
}

function isPromptTool() {
  return state.tool === "include" || state.tool === "exclude";
}

function updatePromptInfo() {
  const here = state.prompts.filter((p) => p.z === state.coords.z);
  const inc = here.filter((p) => p.label === 1).length;
  els.promptInfo.textContent = state.prompts.length
    ? `This slice: ${inc} include, ${here.length - inc} exclude · total ${state.prompts.length}`
    : "No points yet";
}

async function runSegmentForSlice(z = state.coords.z) {
  if (!state.filename) return;
  const payload = promptsForSlice(z);
  if (payload.points.length === 0) {
    delete state.masks[z];
    if (state.useGL) axialView.drawOverlay(state);
    updatePromptInfo();
    return;
  }

  els.promptInfo.textContent = "Segmenting...";
  try {
    const res = await fetch(`${API_BASE}/volumes/${encodeURIComponent(state.filename)}/segment`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const img = new Image();
    img.onload = () => {
      state.masks[z] = img;
      if (state.useGL) axialView.drawOverlay(state);
    };
    img.src = `data:image/png;base64,${data.mask_png_base64}`;

    const hit = data.cache_hit ? "cached" : `encode ${data.encode_ms}ms`;
    els.promptInfo.textContent = `IoU ${data.score.toFixed(2)} · ${data.predict_ms}ms (${hit})`;
  } catch (err) {
    console.error("Segment failed:", err);
    els.promptInfo.textContent = "Segment error (check console)";
  }
}

function addPrompt(event) {
  const v = axialView.eventToVoxel(event, state.coords);
  let label = state.tool === "include" ? 1 : 0;
  if (event.button === 2) label = 1 - label; // right-click adds the opposite point
  state.prompts.push({ x: v.x, y: v.y, z: v.z, label });
  updatePromptInfo();
  els.coordReadout.textContent = `Voxel: (${v.x}, ${v.y}, ${v.z}) · ${label ? "include" : "exclude"}`;
  axialView.drawOverlay(state);
  runSegmentForSlice(v.z);
}

function removeLastPrompt() {
  const removed = state.prompts.pop();
  updatePromptInfo();
  if (state.useGL) axialView.drawOverlay(state);
  if (removed) runSegmentForSlice(removed.z);
}

// The points of one axial slice in the format agreed for the backend:
// { slice, points: [[x, y], ...], labels: [1, 0, ...] } in voxel coordinates.
function promptsForSlice(z = state.coords.z) {
  const here = state.prompts.filter((p) => p.z === z);
  return { slice: z, points: here.map((p) => [p.x, p.y]), labels: here.map((p) => p.label) };
}

// For checking in the browser console: medsam.promptsForSlice()
window.medsam = { state, promptsForSlice };

// --- Three.js 3D Surface Viewer ---
const three = {
  scene: new THREE.Scene(),
  camera: null,
  renderer: null,
  controls: null,
  surfaceMesh: null,
  maskMesh: null,
  crosshairGroup: new THREE.Group(),
};

function initThree() {
  const container = els.threeContainer;
  const w = Math.max(container.clientWidth, 300);
  const h = Math.max(container.clientHeight, 240);

  three.camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 2000);
  three.camera.position.set(180, -180, 140);
  three.camera.up.set(0, 0, 1);

  three.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  three.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  three.renderer.setSize(w, h);
  container.appendChild(three.renderer.domElement);

  three.controls = new OrbitControls(three.camera, three.renderer.domElement);
  three.controls.enableDamping = true;
  three.controls.dampingFactor = 0.08;

  const hemi = new THREE.HemisphereLight(0xeef6ff, 0x222933, 0.9);
  const key = new THREE.DirectionalLight(0xffffff, 1.1);
  key.position.set(150, -200, 220);
  const rim = new THREE.DirectionalLight(0x7dd3fc, 0.45);
  rim.position.set(-180, 180, -100);
  three.scene.add(hemi, key, rim, three.crosshairGroup);

  const ro = new ResizeObserver(() => {
    const cw = container.clientWidth;
    const ch = container.clientHeight;
    if (cw > 0 && ch > 0) {
      three.camera.aspect = cw / ch;
      three.camera.updateProjectionMatrix();
      three.renderer.setSize(cw, ch);
    }
  });
  ro.observe(container);

  function animate() {
    requestAnimationFrame(animate);
    three.controls.update();
    three.renderer.render(three.scene, three.camera);
  }
  animate();
}

function update3DCrosshair() {
  if (!state.info) return;
  while (three.crosshairGroup.children.length > 0) {
    const obj = three.crosshairGroup.children.pop();
    obj.geometry?.dispose();
    obj.material?.dispose();
  }

  const [nx, ny, nz] = state.info.shape;
  const [sx, sy, sz] = state.info.spacing;
  const halfX = (nx - 1) * sx * 0.5;
  const halfY = (ny - 1) * sy * 0.5;
  const halfZ = (nz - 1) * sz * 0.5;

  const px = state.coords.x * sx - halfX;
  const py = state.coords.y * sy - halfY;
  const pz = state.coords.z * sz - halfZ;

  const lineMat = new THREE.LineBasicMaterial({
    color: 0xef4444,
    transparent: true,
    opacity: 0.75,
  });

  const axes = [
    [new THREE.Vector3(-halfX, py, pz), new THREE.Vector3(halfX, py, pz)],
    [new THREE.Vector3(px, -halfY, pz), new THREE.Vector3(px, halfY, pz)],
    [new THREE.Vector3(px, py, -halfZ), new THREE.Vector3(px, py, halfZ)],
  ];
  for (const pts of axes) {
    const geom = new THREE.BufferGeometry().setFromPoints(pts);
    three.crosshairGroup.add(new THREE.Line(geom, lineMat));
  }
}

async function refresh3DMesh(threshold = null) {
  if (!state.filename) return;
  els.meshStatusLabel.textContent = "Reconstructing...";
  const q = threshold !== null ? `?threshold=${encodeURIComponent(threshold)}` : "";
  const res = await fetch(`${API_BASE}/volumes/${encodeURIComponent(state.filename)}/mesh${q}`);
  const data = await res.json();

  state.threshold = data.threshold;
  els.thresholdSlider.value = Math.round(data.threshold);
  els.thresholdVal.textContent = Math.round(data.threshold);

  if (three.surfaceMesh) {
    three.scene.remove(three.surfaceMesh);
    three.surfaceMesh.geometry.dispose();
    three.surfaceMesh.material.dispose();
    three.surfaceMesh = null;
  }

  if (data.face_count === 0) {
    els.meshStats.textContent = "0 tris";
    els.meshStatusLabel.textContent = "Empty (adjust threshold)";
    return;
  }

  const geom = new THREE.BufferGeometry();
  geom.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(data.vertices), 3)
  );
  geom.setIndex(new THREE.BufferAttribute(new Uint32Array(data.faces), 1));
  geom.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: 0xd6cfc2,
    roughness: 0.55,
    metalness: 0.05,
    side: THREE.DoubleSide,
    wireframe: state.wireframe,
  });

  three.surfaceMesh = new THREE.Mesh(geom, mat);
  three.scene.add(three.surfaceMesh);

  // Fit camera distance nicely to bounding sphere on first load
  geom.computeBoundingSphere();
  if (geom.boundingSphere && three.camera) {
    const r = Math.max(geom.boundingSphere.radius, 10);
    const dist = r * 2.5;
    if (threshold === null) {
      three.camera.position.set(dist * 0.75, -dist * 0.75, dist * 0.55);
      three.controls.target.set(0, 0, 0);
      three.controls.update();
    }
  }

  els.meshStats.textContent = `${(data.face_count / 1000).toFixed(1)}k tris`;
  els.meshStatusLabel.textContent = `Threshold = ${Math.round(data.threshold)}`;
}

function setMaskMesh3D(meshData) {
  if (three.maskMesh) {
    three.scene.remove(three.maskMesh);
    three.maskMesh.geometry.dispose();
    three.maskMesh.material.dispose();
    three.maskMesh = null;
  }

  const hasMask = meshData && meshData.face_count > 0;
  if (three.surfaceMesh) {
    three.surfaceMesh.material.transparent = hasMask;
    three.surfaceMesh.material.opacity = hasMask ? 0.2 : 1.0;
    three.surfaceMesh.material.depthWrite = !hasMask;
    three.surfaceMesh.material.needsUpdate = true;
  }
  if (!hasMask) return;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(meshData.vertices), 3));
  geom.setIndex(new THREE.BufferAttribute(new Uint32Array(meshData.faces), 1));
  geom.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    color: 0x38bdf8,
    roughness: 0.4,
    metalness: 0.05,
    side: THREE.DoubleSide,
    wireframe: state.wireframe,
  });
  three.maskMesh = new THREE.Mesh(geom, mat);
  three.scene.add(three.maskMesh);
}

// --- 2D Slice Viewer ---
async function init() {
  initThree();
  const res = await fetch(`${API_BASE}/volumes`);
  const data = await res.json();
  els.volumeSelect.innerHTML = "";
  for (const name of data.volumes) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    els.volumeSelect.appendChild(opt);
  }
  if (data.volumes.length > 0) {
    await loadVolume(data.volumes[0]);
  }
}

async function loadVolume(filename) {
  state.filename = filename;
  const res = await fetch(`${API_BASE}/volumes/${encodeURIComponent(filename)}/info`);
  const info = await res.json();
  state.info = info;

  const [nx, ny, nz] = info.shape;
  state.coords = {
    x: Math.floor(nx / 2),
    y: Math.floor(ny / 2),
    z: Math.floor(nz / 2),
  };

  const span = Math.max(info.max_val - info.min_val, 1);
  els.windowSlider.min = 1;
  els.windowSlider.max = Math.ceil(span * 1.5);
  els.levelSlider.min = Math.floor(info.min_val);
  els.levelSlider.max = Math.ceil(info.max_val);

  els.thresholdSlider.min = Math.floor(info.min_val) + 1;
  els.thresholdSlider.max = Math.ceil(info.max_val) - 1;

  state.window = info.default_window;
  state.level = info.default_level;

  els.sagittalSlider.max = nx - 1;
  els.coronalSlider.max = ny - 1;
  els.axialSlider.max = nz - 1;

  els.metaShape.textContent = `${nx} × ${ny} × ${nz}`;
  els.metaSpacing.textContent = info.spacing.map((s) => s.toFixed(2)).join(" × ");
  els.metaRange.textContent = `${info.min_val.toFixed(0)} ~ ${info.max_val.toFixed(0)}`;

  // Points and masks belong to one volume; start empty
  state.prompts = [];
  state.masks = {};
  setMaskMesh3D(null);

  // Whole volume into the GPU once (server /data, else frontend/dev/<name>.bin)
  let volume = null;
  if (axialView) {
    els.metaRender.textContent = "loading volume...";
    volume = await loadVolumeData(filename, info);
    if (volume) {
      try {
        axialView.setVolume(volume.data, info.shape, info.spacing);
      } catch (err) {
        console.warn(err);
        volume = null;
      }
    }
  }
  if (state.filename !== filename) return; // another volume was picked meanwhile
  setRenderMode(Boolean(volume), volume?.source);

  syncUI();
  refreshAllPlanes();
  update3DCrosshair();
  refresh3DMesh(null);
}

function syncUI() {
  const { x, y, z } = state.coords;
  els.coordReadout.textContent = `Voxel: (${x}, ${y}, ${z})`;

  els.sagittalSlider.value = x;
  els.coronalSlider.value = y;
  els.axialSlider.value = z;

  els.sagittalVal.textContent = x;
  els.coronalVal.textContent = y;
  els.axialVal.textContent = z;

  els.sagittalLabel.textContent = `x = ${x}`;
  els.coronalLabel.textContent = `y = ${y}`;
  els.axialLabel.textContent = `z = ${z}`;

  els.windowSlider.value = Math.round(state.window);
  els.levelSlider.value = Math.round(state.level);
  els.windowVal.textContent = Math.round(state.window);
  els.levelVal.textContent = Math.round(state.level);

  updatePromptInfo();
}

function getSliceIndex(plane) {
  if (plane === "axial") return state.coords.z;
  if (plane === "coronal") return state.coords.y;
  return state.coords.x;
}

function fetchPlane(plane) {
  if (!state.filename) return;
  if (plane === "axial" && state.useGL) {
    axialView.render(state); // drawn in the browser, no server request
    return;
  }
  const idx = getSliceIndex(plane);
  const url =
    `${API_BASE}/volumes/${encodeURIComponent(state.filename)}/slice` +
    `?plane=${plane}&index=${idx}&window=${state.window}&level=${state.level}`;

  const img = new Image();
  img.onload = () => {
    state.images[plane] = img;
    drawPlane(plane);
  };
  img.src = url;
}

function refreshAllPlanes() {
  fetchPlane("axial");
  fetchPlane("coronal");
  fetchPlane("sagittal");
}

function drawPlane(plane) {
  if (plane === "axial" && state.useGL) {
    axialView.drawOverlay(state); // crosshair and points only; the image is WebGL
    return;
  }
  const canvas = els.canvases[plane];
  const img = state.images[plane];
  if (!img || !state.info) return;

  if (canvas.width !== img.width || canvas.height !== img.height) {
    canvas.width = img.width;
    canvas.height = img.height;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(img, 0, 0);

  const [nx, ny, nz] = state.info.shape;
  const { x, y, z } = state.coords;
  let cx = 0;
  let cy = 0;
  if (plane === "axial") {
    cx = x;
    cy = ny - 1 - y;
  } else if (plane === "coronal") {
    cx = x;
    cy = nz - 1 - z;
  } else {
    cx = y;
    cy = nz - 1 - z;
  }

  ctx.save();
  ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(cx + 0.5, 0);
  ctx.lineTo(cx + 0.5, canvas.height);
  ctx.moveTo(0, cy + 0.5);
  ctx.lineTo(canvas.width, cy + 0.5);
  ctx.stroke();
  ctx.restore();
}

function redrawCrosshairs() {
  drawPlane("axial");
  drawPlane("coronal");
  drawPlane("sagittal");
  update3DCrosshair();
}

function handleCanvasPointer(plane, event) {
  if (!state.info) return;
  if (plane === "axial" && state.useGL) {
    // The overlay is drawn at screen resolution, so convert through the view
    const v = axialView.eventToVoxel(event, state.coords);
    state.coords.x = v.x;
    state.coords.y = v.y;
    fetchPlane("coronal");
    fetchPlane("sagittal");
    axialView.drawOverlay(state);
    syncUI();
    update3DCrosshair();
    return;
  }
  const canvas = els.canvases[plane];
  const rect = canvas.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;

  const px = clamp(
    Math.floor(((event.clientX - rect.left) / rect.width) * canvas.width),
    0,
    canvas.width - 1
  );
  const py = clamp(
    Math.floor(((event.clientY - rect.top) / rect.height) * canvas.height),
    0,
    canvas.height - 1
  );

  const [nx, ny, nz] = state.info.shape;
  if (plane === "axial") {
    state.coords.x = clamp(px, 0, nx - 1);
    state.coords.y = clamp(ny - 1 - py, 0, ny - 1);
    fetchPlane("coronal");
    fetchPlane("sagittal");
    drawPlane("axial");
  } else if (plane === "coronal") {
    state.coords.x = clamp(px, 0, nx - 1);
    state.coords.z = clamp(nz - 1 - py, 0, nz - 1);
    fetchPlane("axial");
    fetchPlane("sagittal");
    drawPlane("coronal");
  } else {
    state.coords.y = clamp(px, 0, ny - 1);
    state.coords.z = clamp(nz - 1 - py, 0, nz - 1);
    fetchPlane("axial");
    fetchPlane("coronal");
    drawPlane("sagittal");
  }
  syncUI();
  update3DCrosshair();
}

// Event bindings
els.volumeSelect.addEventListener("change", (e) => loadVolume(e.target.value));

els.axialSlider.addEventListener("input", (e) => {
  state.coords.z = Number(e.target.value);
  syncUI();
  fetchPlane("axial");
  redrawCrosshairs();
});
els.coronalSlider.addEventListener("input", (e) => {
  state.coords.y = Number(e.target.value);
  syncUI();
  fetchPlane("coronal");
  redrawCrosshairs();
});
els.sagittalSlider.addEventListener("input", (e) => {
  state.coords.x = Number(e.target.value);
  syncUI();
  fetchPlane("sagittal");
  redrawCrosshairs();
});

let wlTimer = null;
function onWLChange() {
  state.window = Number(els.windowSlider.value);
  state.level = Number(els.levelSlider.value);
  syncUI();
  if (state.useGL) axialView.render(state); // instant: only shader values change
  clearTimeout(wlTimer);
  wlTimer = setTimeout(refreshAllPlanes, 30);
}
els.windowSlider.addEventListener("input", onWLChange);
els.levelSlider.addEventListener("input", onWLChange);

els.resetWLBtn.addEventListener("click", () => {
  if (!state.info) return;
  state.window = state.info.default_window;
  state.level = state.info.default_level;
  syncUI();
  refreshAllPlanes();
});

els.thresholdSlider.addEventListener("input", (e) => {
  els.thresholdVal.textContent = e.target.value;
});
els.thresholdSlider.addEventListener("change", (e) => {
  refresh3DMesh(Number(e.target.value));
});
els.wireframeToggle.addEventListener("change", (e) => {
  state.wireframe = e.target.checked;
  if (three.surfaceMesh) {
    three.surfaceMesh.material.wireframe = state.wireframe;
  }
});

for (const plane of ["axial", "coronal", "sagittal"]) {
  const canvas = els.canvases[plane];
  let dragging = false;

  canvas.addEventListener("mousedown", (e) => {
    if (plane === "axial" && state.useGL && isPromptTool()) {
      addPrompt(e); // one point per click, no dragging
      return;
    }
    dragging = true;
    handleCanvasPointer(plane, e);
  });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  window.addEventListener("mouseup", () => {
    dragging = false;
  });
  canvas.addEventListener("mousemove", (e) => {
    if (dragging) handleCanvasPointer(plane, e);
  });

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      if (!state.info) return;
      const [nx, ny, nz] = state.info.shape;
      const delta = e.deltaY > 0 ? 1 : -1;
      if (plane === "axial") {
        state.coords.z = clamp(state.coords.z + delta, 0, nz - 1);
      } else if (plane === "coronal") {
        state.coords.y = clamp(state.coords.y + delta, 0, ny - 1);
      } else {
        state.coords.x = clamp(state.coords.x + delta, 0, nx - 1);
      }
      syncUI();
      fetchPlane(plane);
      redrawCrosshairs();
    },
    { passive: false }
  );
}

// Tools: buttons, keys N / I / E, Ctrl+Z to undo the last point
for (const btn of els.toolButtons) {
  btn.addEventListener("click", () => setTool(btn.dataset.tool));
}
async function runPropagateFromSlice(z = state.coords.z) {
  if (!state.filename) return;
  const payload = promptsForSlice(z);
  if (payload.points.length === 0) {
    els.promptInfo.textContent = "Add a point on this slice first";
    return;
  }

  els.promptInfo.textContent = "Propagating both directions...";
  try {
    const res = await fetch(
      `${API_BASE}/volumes/${encodeURIComponent(state.filename)}/propagate?max_steps=15`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const entries = Object.entries(data.slices);
    await Promise.all(
      entries.map(
        ([sliceKey, b64]) =>
          new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
              state.masks[Number(sliceKey)] = img;
              resolve();
            };
            img.src = `data:image/png;base64,${b64}`;
          })
      )
    );
    if (state.useGL) axialView.drawOverlay(state);
    setMaskMesh3D(data.mask_mesh);
    const [zMin, zMax] = data.slice_range ?? [z, z];
    els.promptInfo.textContent =
      `z ${zMin}–${zMax} (${data.slice_count} slices) · ${data.volume_ml} mL · ${data.elapsed_ms}ms`;
  } catch (err) {
    console.error("Propagate failed:", err);
    els.promptInfo.textContent = "Propagate error (check console)";
  }
}

els.propagateBtn?.addEventListener("click", () => runPropagateFromSlice(state.coords.z));
els.undoPromptBtn.addEventListener("click", removeLastPrompt);
els.clearPromptsBtn.addEventListener("click", () => {
  state.prompts = [];
  state.masks = {};
  setMaskMesh3D(null);
  updatePromptInfo();
  if (state.useGL) axialView.drawOverlay(state);
});
window.addEventListener("keydown", (e) => {
  if (e.target instanceof Element && e.target.closest("input, select, textarea")) return;
  const key = e.key.toLowerCase();
  if ((e.ctrlKey || e.metaKey) && key === "z") {
    e.preventDefault();
    removeLastPrompt();
  } else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
    if (key === "n" || key === "escape") setTool("navigate");
    else if (key === "i") setTool("include");
    else if (key === "e") setTool("exclude");
  }
});
setTool("navigate");

init();
