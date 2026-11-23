// One 2D slice view drawn with WebGL2 straight from the volume in GPU memory.
//
// Two canvases are stacked in the same spot:
//   glCanvas      : WebGL2, draws the grey image (window/level in the shader)
//   overlayCanvas : 2D, on top, draws the crosshair and the click points and
//                   receives the mouse events
// A canvas can have either a WebGL or a 2D context, never both, hence two.
//
// The volume is uploaded once as a 3D texture. Scrolling and window/level only
// change shader uniforms, so they never wait on the server.
//
// Screen layout matches the backend's PNG slices (np.rot90): for the axial view
// columns are x (left to right) and rows are y with the highest y at the top.

const VERTEX_SHADER = `#version 300 es
// One triangle that covers the whole canvas; no vertex buffer needed.
out vec2 vUV; // (0,0) = top-left of the view, (1,1) = bottom-right
void main() {
  vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)); // (0,0) (2,0) (0,2)
  vUV = vec2(pos.x, 1.0 - pos.y); // flip y: WebGL's y points up, screen rows point down
  gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler3D;

uniform sampler3D uVolume;  // R32F, one float per voxel
uniform ivec3 uShape;       // nx, ny, nz
uniform int uPlane;         // 0 axial, 1 coronal, 2 sagittal
uniform int uIndex;         // slice number in that plane
uniform float uWindow;
uniform float uLevel;

in vec2 vUV;
out vec4 outColor;

void main() {
  ivec3 v;
  if (uPlane == 0) {        // axial: columns = x, rows = y (high y at the top)
    v = ivec3(int(vUV.x * float(uShape.x)), uShape.y - 1 - int(vUV.y * float(uShape.y)), uIndex);
  } else if (uPlane == 1) { // coronal: columns = x, rows = z (high z at the top)
    v = ivec3(int(vUV.x * float(uShape.x)), uIndex, uShape.z - 1 - int(vUV.y * float(uShape.z)));
  } else {                  // sagittal: columns = y, rows = z (high z at the top)
    v = ivec3(uIndex, int(vUV.x * float(uShape.y)), uShape.z - 1 - int(vUV.y * float(uShape.z)));
  }
  v = clamp(v, ivec3(0), uShape - 1);

  float value = texelFetch(uVolume, v, 0).r;
  // Same formula as the backend's apply_window_level
  float grey = clamp((value - (uLevel - 0.5 * uWindow)) / max(uWindow, 1e-6), 0.0, 1.0);
  outColor = vec4(vec3(grey), 1.0);
}`;

const PLANES = { axial: 0, coronal: 1, sagittal: 2 };
const INCLUDE_COLOR = "#22c55e";
const EXCLUDE_COLOR = "#ef4444";

function compile(gl, type, source) {
  const shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(shader));
  }
  return shader;
}

export class SliceView {
  constructor(glCanvas, overlayCanvas, plane) {
    const gl = glCanvas.getContext("webgl2", { antialias: false });
    if (!gl) throw new Error("WebGL2 is not available in this browser");

    this.gl = gl;
    this.glCanvas = glCanvas;
    this.overlay = overlayCanvas;
    this.stack = glCanvas.parentElement; // .slice-stack around both canvases
    this.plane = plane;
    this.shape = null;
    this.spacing = null;
    this.active = false; // false while the app shows PNG slices instead

    const program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, VERTEX_SHADER));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
    }
    this.program = program;
    this.uniforms = {};
    for (const name of ["uVolume", "uShape", "uPlane", "uIndex", "uWindow", "uLevel"]) {
      this.uniforms[name] = gl.getUniformLocation(program, name);
    }
    this.vao = gl.createVertexArray(); // empty, but WebGL2 wants one bound to draw
    this.texture = null;

    // Re-fit when the panel changes size
    this.lastState = null;
    new ResizeObserver(() => {
      if (this.active) {
        this.resize();
        if (this.lastState) this.render(this.lastState);
      }
    }).observe(this.stack.parentElement);
  }

  // Upload the whole volume once (Float32Array, x fastest).
  setVolume(data, shape, spacing) {
    const gl = this.gl;
    const [nx, ny, nz] = shape;
    const max = gl.getParameter(gl.MAX_3D_TEXTURE_SIZE);
    if (Math.max(nx, ny, nz) > max) {
      throw new Error(`Volume ${nx}×${ny}×${nz} is larger than this GPU's 3D texture limit (${max})`);
    }
    if (this.texture) gl.deleteTexture(this.texture);
    this.texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_3D, this.texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(gl.TEXTURE_3D, 0, gl.R32F, nx, ny, nz, 0, gl.RED, gl.FLOAT, data);
    // Float textures cannot be filtered linearly without an extension; we read exact voxels anyway.
    for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_3D, p, gl.NEAREST);
    for (const p of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T, gl.TEXTURE_WRAP_R]) {
      gl.texParameteri(gl.TEXTURE_3D, p, gl.CLAMP_TO_EDGE);
    }
    this.shape = shape;
    this.spacing = spacing;
    this.active = true;
    this.resize();
  }

  // Stop drawing (the app went back to PNG slices for this volume).
  deactivate() {
    this.active = false;
    this.lastState = null;
  }

  // Columns and rows of this view, in voxels and in mm per voxel.
  layout() {
    const [nx, ny, nz] = this.shape;
    const [sx, sy, sz] = this.spacing;
    if (this.plane === "axial") return { cols: nx, rows: ny, colMm: sx, rowMm: sy };
    if (this.plane === "coronal") return { cols: nx, rows: nz, colMm: sx, rowMm: sz };
    return { cols: ny, rows: nz, colMm: sy, rowMm: sz };
  }

  sliceIndex(coords) {
    if (this.plane === "axial") return coords.z;
    if (this.plane === "coronal") return coords.y;
    return coords.x;
  }

  // Voxel -> (column, row) on screen, and back.
  toScreen(v) {
    const [, ny, nz] = this.shape;
    if (this.plane === "axial") return { col: v.x, row: ny - 1 - v.y };
    if (this.plane === "coronal") return { col: v.x, row: nz - 1 - v.z };
    return { col: v.y, row: nz - 1 - v.z };
  }

  toVoxel(col, row, coords) {
    const [, ny, nz] = this.shape;
    if (this.plane === "axial") return { x: col, y: ny - 1 - row, z: coords.z };
    if (this.plane === "coronal") return { x: col, y: coords.y, z: nz - 1 - row };
    return { x: coords.x, y: col, z: nz - 1 - row };
  }

  // Fill the panel while keeping the true proportions (voxels × spacing).
  resize() {
    const wrap = this.stack.parentElement;
    const style = getComputedStyle(wrap);
    const availW = wrap.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const availH = wrap.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    if (availW <= 0 || availH <= 0) return;

    const { cols, rows, colMm, rowMm } = this.layout();
    const widthMm = cols * colMm;
    const heightMm = rows * rowMm;
    const scale = Math.min(availW / widthMm, availH / heightMm);
    const cssW = Math.max(1, Math.floor(widthMm * scale));
    const cssH = Math.max(1, Math.floor(heightMm * scale));

    this.stack.style.width = `${cssW}px`;
    this.stack.style.height = `${cssH}px`;
    const dpr = window.devicePixelRatio || 1;
    for (const canvas of [this.glCanvas, this.overlay]) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
  }

  // Draw the grey image, then the crosshair and points on top.
  render(state) {
    if (!this.texture) return;
    this.lastState = state;
    const gl = this.gl;
    const u = this.uniforms;
    gl.viewport(0, 0, this.glCanvas.width, this.glCanvas.height);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, this.texture);
    gl.uniform1i(u.uVolume, 0);
    gl.uniform3i(u.uShape, ...this.shape);
    gl.uniform1i(u.uPlane, PLANES[this.plane]);
    gl.uniform1i(u.uIndex, this.sliceIndex(state.coords));
    gl.uniform1f(u.uWindow, state.window);
    gl.uniform1f(u.uLevel, state.level);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    this.drawOverlay(state);
  }

  drawOverlay(state) {
    if (!this.shape) return;
    const ctx = this.overlay.getContext("2d");
    const { width, height } = this.overlay;
    const { cols, rows } = this.layout();
    const px = (col) => ((col + 0.5) / cols) * width;
    const py = (row) => ((row + 0.5) / rows) * height;
    const dpr = window.devicePixelRatio || 1;
    ctx.clearRect(0, 0, width, height);

    // Segmentation mask overlay for this slice (if available)
    const index = this.sliceIndex(state.coords);
    const maskImg = state.masks?.[index];
    if (maskImg) {
      ctx.drawImage(maskImg, 0, 0, width, height);
    }

    // Crosshair
    const c = this.toScreen(state.coords);
    ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
    ctx.lineWidth = dpr;
    ctx.beginPath();
    ctx.moveTo(Math.round(px(c.col)) + 0.5, 0);
    ctx.lineTo(Math.round(px(c.col)) + 0.5, height);
    ctx.moveTo(0, Math.round(py(c.row)) + 0.5);
    ctx.lineTo(width, Math.round(py(c.row)) + 0.5);
    ctx.stroke();

    // Points placed on this slice only
    for (const p of state.prompts ?? []) {
      if (this.sliceIndex(p) !== index) continue;
      const s = this.toScreen(p);
      ctx.beginPath();
      ctx.arc(px(s.col), py(s.row), 4.5 * dpr, 0, 2 * Math.PI);
      ctx.fillStyle = p.label === 1 ? INCLUDE_COLOR : EXCLUDE_COLOR;
      ctx.fill();
      ctx.lineWidth = 1.5 * dpr;
      ctx.strokeStyle = "#ffffff";
      ctx.stroke();
    }
  }

  // Mouse position -> voxel (x, y, z) on the current slice.
  eventToVoxel(event, coords) {
    const rect = this.overlay.getBoundingClientRect();
    const { cols, rows } = this.layout();
    const col = Math.min(cols - 1, Math.max(0, Math.floor(((event.clientX - rect.left) / rect.width) * cols)));
    const row = Math.min(rows - 1, Math.max(0, Math.floor(((event.clientY - rect.top) / rect.height) * rows)));
    return this.toVoxel(col, row, coords);
  }
}
