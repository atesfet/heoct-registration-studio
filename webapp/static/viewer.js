"use strict";
// Interactive registration viewer: pans/zooms over the OCT pixel grid, fetches only the visible region of each
// layer (server-side pyramids), and composites H&E and OCT client-side (blend / swipe / checker / flicker /
// false colour / side by side) so every slider responds instantly.
(() => {
  const $ = (id) => document.getElementById(id);
  const V = {
    r: null, dir: null, W: 0, H: 0, px: 2, layers: [],
    cx: 0, cy: 0, z: 1, mode: "blend", xform: "def", alpha: 0.5, swipe: 0.5, tileUm: 100, invert: false, flickMs: 600, flickOn: true,
    imgs: {}, req: 0, timer: null, flickT: null, mouse: null, pins: [], drag: null, mini: {},
  };
  const cv = $("view"), ctx = cv.getContext("2d");
  const dpr = () => window.devicePixelRatio || 1;

  // ------------------------------------------------------------ layers needed for the current state
  function heLayer() {
    if (V.xform === "aff" && V.layers.includes("he_affine")) return "he_affine";
    return $("orig-colours").checked && V.layers.includes("he_orig") ? "he_orig" : "he";
  }
  function needed() {
    const L = new Set();
    if (V.mode !== "oct") L.add(V.mode === "false" ? (V.xform === "aff" ? "he_od_affine" : "he_od") : heLayer());
    if (V.mode !== "he") L.add("oct");
    if ($("ov-contour").checked) L.add(V.xform === "aff" && V.layers.includes("contour_affine") ? "contour_affine" : "contour");
    return [...L].filter((l) => V.layers.includes(l));
  }

  // ------------------------------------------------------------ geometry
  function size() {
    const r = cv.getBoundingClientRect(); const w = Math.max(10, Math.round(r.width * dpr())), h = Math.max(10, Math.round(r.height * dpr()));
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    return { w, h };
  }
  function panels() {   // screen rectangles (device px) each showing the view centred at (cx, cy)
    const { w, h } = size();
    if (V.mode === "side") { const hw = Math.floor(w / 2); return [{ x: 0, y: 0, w: hw - 1, h, role: "he" }, { x: hw + 1, y: 0, w: w - hw - 1, h, role: "oct" }]; }
    return [{ x: 0, y: 0, w, h, role: "all" }];
  }
  const zd = () => V.z * dpr();                       // device px per OCT px
  function region(p) { const s = zd(); return { x: V.cx - p.w / 2 / s, y: V.cy - p.h / 2 / s, w: p.w / s, h: p.h / s }; }
  function toScreen(p, X, Y) { const s = zd(), R = region(p); return [p.x + (X - R.x) * s, p.y + (Y - R.y) * s]; }
  function toImage(clientX, clientY) {
    const b = cv.getBoundingClientRect(), dx = (clientX - b.left) * dpr(), dy = (clientY - b.top) * dpr();
    const P = panels(); const p = P.find((q) => dx >= q.x && dx <= q.x + q.w) || P[0];
    const R = region(p), s = zd(); return { X: R.x + (dx - p.x) / s, Y: R.y + (dy - p.y) / s, p, dx, dy };
  }

  // ------------------------------------------------------------ fetching
  function schedule(delay = 70) { clearTimeout(V.timer); V.timer = setTimeout(fetchAll, delay); draw(); }
  function fetchAll() {
    if (!V.dir) return;
    const p = panels()[0]; const R = region(p); const id = ++V.req;
    const ow = Math.min(4096, p.w), oh = Math.min(4096, p.h);
    const L = needed(); let pending = L.length; $("vbusy").classList.toggle("hidden", !pending);
    for (const layer of L) {
      const url = `/api/tile?dir=${encodeURIComponent(V.dir)}&layer=${layer}&x=${R.x.toFixed(2)}&y=${R.y.toFixed(2)}&w=${R.w.toFixed(2)}&h=${R.h.toFixed(2)}&ow=${ow}&oh=${oh}`;
      const img = new Image();
      img.onload = () => { if (id >= (V.imgs[layer]?.id || 0)) V.imgs[layer] = { img, R, id }; if (--pending <= 0 && id === V.req) $("vbusy").classList.add("hidden"); draw(); };
      img.onerror = () => { if (--pending <= 0) $("vbusy").classList.add("hidden"); };
      img.src = url;
    }
  }
  function blit(c, layer, p) {   // draw a fetched layer image into panel p, transformed to the current view
    const e = V.imgs[layer]; if (!e) return false;
    const [x0, y0] = toScreen(p, e.R.x, e.R.y), s = zd();
    c.drawImage(e.img, x0, y0, e.R.w * s, e.R.h * s); return true;
  }

  // ------------------------------------------------------------ compositing
  const off = document.createElement("canvas"), offc = off.getContext("2d");
  const off2 = document.createElement("canvas"), off2c = off2.getContext("2d");
  function tinted(oc, layer, p, col) {  // layer * colour into an offscreen canvas
    oc.canvas.width = cv.width; oc.canvas.height = cv.height;
    oc.globalCompositeOperation = "source-over"; oc.fillStyle = "#000"; oc.fillRect(0, 0, cv.width, cv.height);
    oc.save(); oc.beginPath(); oc.rect(p.x, p.y, p.w, p.h); oc.clip(); blit(oc, layer, p); oc.restore();
    oc.globalCompositeOperation = "multiply"; oc.fillStyle = col; oc.fillRect(0, 0, cv.width, cv.height);
    oc.globalCompositeOperation = "source-over";
  }
  function checkerClip(c, p) {
    const t = V.tileUm / V.px, R = region(p), s = zd();
    const i0 = Math.floor(R.x / t), i1 = Math.ceil((R.x + R.w) / t), j0 = Math.floor(R.y / t), j1 = Math.ceil((R.y + R.h) / t);
    c.beginPath();
    for (let j = j0; j < j1; j++) for (let i = i0; i < i1; i++) {
      if (((i + j) & 1) === (V.invert ? 1 : 0)) { const [x, y] = toScreen(p, i * t, j * t); c.rect(x, y, t * s + 0.5, t * s + 0.5); }
    }
    c.clip();
  }
  function draw() {
    if (!V.dir) return;
    size(); ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.fillStyle = "#000"; ctx.fillRect(0, 0, cv.width, cv.height);
    const he = V.mode === "false" ? (V.xform === "aff" ? "he_od_affine" : "he_od") : heLayer();
    for (const p of panels()) {
      ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w, p.h); ctx.clip();
      const m = V.mode;
      if (m === "side") blit(ctx, p.role === "he" ? he : "oct", p);
      else if (m === "he") blit(ctx, he, p);
      else if (m === "oct") blit(ctx, "oct", p);
      else if (m === "flicker") blit(ctx, V.flickOn ? he : "oct", p);
      else if (m === "blend") { blit(ctx, "oct", p); ctx.globalAlpha = V.alpha; blit(ctx, he, p); ctx.globalAlpha = 1; }
      else if (m === "swipe") { blit(ctx, "oct", p); ctx.save(); ctx.beginPath(); ctx.rect(p.x, p.y, p.w * V.swipe, p.h); ctx.clip(); blit(ctx, he, p); ctx.restore();
        ctx.strokeStyle = "#ffd60a"; ctx.lineWidth = 2 * dpr(); ctx.beginPath(); ctx.moveTo(p.x + p.w * V.swipe, p.y); ctx.lineTo(p.x + p.w * V.swipe, p.y + p.h); ctx.stroke(); }
      else if (m === "checker") { blit(ctx, "oct", p); ctx.save(); checkerClip(ctx, p); blit(ctx, he, p); ctx.restore(); }
      else if (m === "false") { tinted(offc, "oct", p, "#ff00ff"); tinted(off2c, he, p, "#00ff00");
        ctx.drawImage(off, 0, 0); ctx.globalCompositeOperation = "lighter"; ctx.drawImage(off2, 0, 0); ctx.globalCompositeOperation = "source-over"; }
      const cl = V.xform === "aff" && V.layers.includes("contour_affine") ? "contour_affine" : "contour";
      if ($("ov-contour").checked) blit(ctx, cl, p);
      if ($("ov-grid").checked) grid(p);
      if ($("ov-cross").checked) marks(p);
      if (m === "side") { ctx.fillStyle = "rgba(0,0,0,.55)"; ctx.fillRect(p.x + 6, p.y + 6, 70 * dpr(), 20 * dpr()); ctx.fillStyle = "#fff";
        ctx.font = `${12 * dpr()}px sans-serif`; ctx.fillText(p.role === "he" ? "H&E" : "OCT", p.x + 12, p.y + 20 * dpr()); }
      ctx.restore();
    }
    scalebar(); minimap();
  }
  function grid(p) {
    const t = 100 / V.px, R = region(p); ctx.strokeStyle = "rgba(255,214,10,.55)"; ctx.lineWidth = 1;
    ctx.beginPath();
    for (let X = Math.ceil(R.x / t) * t; X < R.x + R.w; X += t) { const [x] = toScreen(p, X, 0); ctx.moveTo(x, p.y); ctx.lineTo(x, p.y + p.h); }
    for (let Y = Math.ceil(R.y / t) * t; Y < R.y + R.h; Y += t) { const [, y] = toScreen(p, 0, Y); ctx.moveTo(p.x, y); ctx.lineTo(p.x + p.w, y); }
    ctx.stroke();
  }
  function marks(p) {
    ctx.lineWidth = 1.5 * dpr();
    if (V.mouse) { const [x, y] = toScreen(p, V.mouse.X, V.mouse.Y); ctx.strokeStyle = "#00e5ff"; ctx.beginPath();
      ctx.moveTo(x - 12 * dpr(), y); ctx.lineTo(x + 12 * dpr(), y); ctx.moveTo(x, y - 12 * dpr()); ctx.lineTo(x, y + 12 * dpr()); ctx.stroke(); }
    V.pins.forEach((q, i) => { const [x, y] = toScreen(p, q.X, q.Y); ctx.fillStyle = "#ff3b3b"; ctx.beginPath(); ctx.arc(x, y, 4 * dpr(), 0, 7); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.font = `bold ${12 * dpr()}px sans-serif`; ctx.fillText(String(i + 1), x + 6 * dpr(), y - 6 * dpr()); });
  }
  function scalebar() {
    const target = 120 / V.z * V.px; const nice = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
    const um = nice.find((n) => n >= target * 0.6) || 5000; const L = um / V.px * V.z;
    $("vscale").innerHTML = `<div class="bar" style="width:${L}px"></div>${um >= 1000 ? um / 1000 + " mm" : um + " µm"} · zoom ${(V.z * V.px).toFixed(2)} css px/µm` +
      (V.z * dpr() >= 1 ? ` · ${(V.z * dpr()).toFixed(1)} screen px per OCT px` : "");
  }

  // ------------------------------------------------------------ minimap
  function loadMini() {
    const W = 240, s = W / Math.max(V.W, V.H), ow = Math.round(V.W * s), oh = Math.round(V.H * s); V.mini = { s, ow, oh };
    for (const layer of ["oct", "he"]) {
      const img = new Image(); img.onload = () => { V.mini[layer] = img; minimap(); };
      img.src = `/api/tile?dir=${encodeURIComponent(V.dir)}&layer=${layer}&x=0&y=0&w=${V.W}&h=${V.H}&ow=${ow}&oh=${oh}`;
    }
  }
  function minimap() {
    const m = $("minimap"), c = m.getContext("2d"), M = V.mini; if (!M.s) return;
    m.width = M.ow; m.height = M.oh; c.fillStyle = "#000"; c.fillRect(0, 0, m.width, m.height);
    if (M.oct) c.drawImage(M.oct, 0, 0); if (M.he) { c.globalAlpha = 0.5; c.drawImage(M.he, 0, 0); c.globalAlpha = 1; }
    const R = region(panels()[0]); c.strokeStyle = "#ffd60a"; c.lineWidth = 2; c.strokeRect(R.x * M.s, R.y * M.s, R.w * M.s, R.h * M.s);
    V.pins.forEach((q) => { c.fillStyle = "#ff3b3b"; c.fillRect(q.X * M.s - 2, q.Y * M.s - 2, 4, 4); });
  }

  // ------------------------------------------------------------ navigation
  function fit() { const p = panels()[0]; V.z = Math.min(p.w / V.W, p.h / V.H) / dpr() * 0.96; V.cx = V.W / 2; V.cy = V.H / 2; schedule(0); }
  function zoomAt(f, X, Y) { const z2 = Math.min(64, Math.max(0.01, V.z * f)); if (X !== undefined) { V.cx = X - (X - V.cx) * V.z / z2; V.cy = Y - (Y - V.cy) * V.z / z2; } V.z = z2; schedule(); }
  function setMode(m) {
    V.mode = m; document.querySelectorAll("#mode button").forEach((b) => b.classList.toggle("on", b.dataset.v === m));
    $("g-alpha").classList.toggle("hidden", m !== "blend"); $("g-swipe").classList.toggle("hidden", m !== "swipe");
    $("g-checker").classList.toggle("hidden", m !== "checker"); $("g-flicker").classList.toggle("hidden", m !== "flicker");
    clearInterval(V.flickT); if (m === "flicker") V.flickT = setInterval(() => { V.flickOn = !V.flickOn; draw(); }, V.flickMs);
    schedule(0);
  }
  function readout(e) {
    const q = toImage(e.clientX, e.clientY); V.mouse = { X: q.X, Y: q.Y };
    const inside = q.X >= 0 && q.Y >= 0 && q.X < V.W && q.Y < V.H;
    $("readout").textContent = inside ? `OCT px  x ${q.X.toFixed(1)}  y ${q.Y.toFixed(1)}\nOCT µm  x ${((q.X + 0.5) * V.px).toFixed(1)}  y ${((q.Y + 0.5) * V.px).toFixed(1)}` : "outside the OCT frame";
  }
  async function addPin(X, Y) {
    if (X < 0 || Y < 0 || X >= V.W || Y >= V.H) return;
    try {
      const c = await fetch(`/api/coord?dir=${encodeURIComponent(V.dir)}&x=${X.toFixed(2)}&y=${Y.toFixed(2)}`).then((r) => r.json());
      V.pins.push({ X, Y, c });
      $("pinout").textContent = V.pins.map((p, i) => `#${i + 1}  OCT px (${p.X.toFixed(1)}, ${p.Y.toFixed(1)})  = (${p.c.um[0].toFixed(1)}, ${p.c.um[1].toFixed(1)}) µm\n` +
        `    scanner (${p.c.scanner_mm[0].toFixed(4)}, ${p.c.scanner_mm[1].toFixed(4)}) mm` + (p.c.he_px ? `\n    H&E native px (${p.c.he_px[0].toFixed(1)}, ${p.c.he_px[1].toFixed(1)})` : "")).join("\n");
      draw();
    } catch (e) { /* ignore */ }
  }

  // ------------------------------------------------------------ events
  cv.addEventListener("wheel", (e) => { e.preventDefault(); const q = toImage(e.clientX, e.clientY); zoomAt(Math.exp(-e.deltaY * 0.0015), q.X, q.Y); }, { passive: false });
  cv.addEventListener("pointerdown", (e) => { cv.setPointerCapture(e.pointerId); V.drag = { x: e.clientX, y: e.clientY, cx: V.cx, cy: V.cy, moved: false, shift: e.shiftKey }; cv.classList.add("drag"); });
  cv.addEventListener("pointermove", (e) => {
    readout(e);
    if (V.drag) {
      const dx = e.clientX - V.drag.x, dy = e.clientY - V.drag.y; if (Math.hypot(dx, dy) > 3) V.drag.moved = true;
      if (V.drag.shift && V.mode === "swipe") { const b = cv.getBoundingClientRect(); V.swipe = Math.min(1, Math.max(0, (e.clientX - b.left) / b.width)); $("swipe").value = Math.round(V.swipe * 100); draw(); }
      else { V.cx = V.drag.cx - dx / V.z; V.cy = V.drag.cy - dy / V.z; schedule(); }
    } else draw();
  });
  cv.addEventListener("pointerup", (e) => { cv.classList.remove("drag"); const d = V.drag; V.drag = null; if (d && !d.moved) { const q = toImage(e.clientX, e.clientY); addPin(q.X, q.Y); } });
  cv.addEventListener("pointerleave", () => { V.mouse = null; draw(); });
  cv.addEventListener("dblclick", (e) => { const q = toImage(e.clientX, e.clientY); V.pins.pop(); zoomAt(2, q.X, q.Y); });
  $("minimap").addEventListener("click", (e) => { const b = e.target.getBoundingClientRect(); V.cx = (e.clientX - b.left) / b.width * V.W; V.cy = (e.clientY - b.top) / b.height * V.H; schedule(0); });
  document.querySelectorAll("#mode button").forEach((b) => (b.onclick = () => setMode(b.dataset.v)));
  document.querySelectorAll("#xform button").forEach((b) => (b.onclick = () => { V.xform = b.dataset.v; document.querySelectorAll("#xform button").forEach((x) => x.classList.toggle("on", x === b)); schedule(0); }));
  $("alpha").oninput = (e) => { V.alpha = e.target.value / 100; $("alpha-v").textContent = e.target.value + "%"; draw(); };
  $("swipe").oninput = (e) => { V.swipe = e.target.value / 100; draw(); };
  $("tile").oninput = (e) => { V.tileUm = +e.target.value; $("tile-v").textContent = e.target.value + " µm"; draw(); };
  $("invert").onchange = (e) => { V.invert = e.target.checked; draw(); };
  $("flick").oninput = (e) => { V.flickMs = +e.target.value; $("flick-v").textContent = e.target.value + " ms"; if (V.mode === "flicker") setMode("flicker"); };
  for (const id of ["ov-contour", "orig-colours"]) $(id).onchange = () => schedule(0);
  for (const id of ["ov-grid", "ov-cross"]) $(id).onchange = draw;
  $("btn-fit").onclick = fit;
  $("btn-1to1").onclick = () => { V.z = 1 / dpr(); schedule(0); };
  $("btn-clear-pins").onclick = () => { V.pins = []; $("pinout").textContent = "click to drop a pin (H&E native pixel lookup)"; draw(); };
  $("btn-png").onclick = () => { const a = document.createElement("a"); a.href = cv.toDataURL("image/png");
    a.download = `${V.r.summary.sample_id}_${V.mode}_x${Math.round(V.cx)}_y${Math.round(V.cy)}.png`; a.click(); };
  $("roi-jump").onchange = (e) => { const r = V.r.rois[+e.target.value]; if (!r) return; V.cx = r.cx_um / V.px; V.cy = r.cy_um / V.px;
    const p = panels()[0]; V.z = Math.min(p.w, p.h) / dpr() / (700 / V.px); schedule(0); };
  window.addEventListener("resize", () => schedule(0));
  document.addEventListener("keydown", (e) => {
    if (!V.dir || $("viewer-wrap").classList.contains("hidden") || /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) return;
    const step = 80 / V.z, modes = ["blend", "swipe", "checker", "flicker", "false", "side", "he", "oct"];
    if (e.key === "+" || e.key === "=") zoomAt(1.4, V.cx, V.cy); else if (e.key === "-") zoomAt(1 / 1.4, V.cx, V.cy);
    else if (e.key === "ArrowLeft") { V.cx -= step; schedule(); } else if (e.key === "ArrowRight") { V.cx += step; schedule(); }
    else if (e.key === "ArrowUp") { V.cy -= step; schedule(); } else if (e.key === "ArrowDown") { V.cy += step; schedule(); }
    else if (/^[1-8]$/.test(e.key)) setMode(modes[+e.key - 1]);
    else if (e.key === "f" || e.key === "F") fit();
    else if (e.key === "i" || e.key === "I") { $("invert").checked = !$("invert").checked; V.invert = $("invert").checked; draw(); }
    else if (e.key === " ") { e.preventDefault(); if (V.mode === "he") setMode("oct"); else if (V.mode === "oct") setMode("he"); else { V.flickOn = !V.flickOn; setMode(V.mode === "flicker" ? "blend" : "flicker"); } }
    else return;
    e.preventDefault();
  });

  // ------------------------------------------------------------ public
  window.RegViewer = {
    open(r) {
      V.r = r; V.dir = r.summary.dir; V.H = r.summary.oct_shape[0]; V.W = r.summary.oct_shape[1]; V.px = r.summary.oct_px_um;
      V.layers = r.layers; V.imgs = {}; V.pins = []; V.mini = {};
      $("roi-jump").innerHTML = '<option value="">Jump to QC ROI…</option>' + (r.rois || []).map((x, i) => `<option value="${i}">ROI ${String(x.id).padStart(2, "0")} (${x.kind}) · ${Math.round(x.cx_um)}, ${Math.round(x.cy_um)} µm</option>`).join("");
      document.querySelector('#xform button[data-v="aff"]').disabled = !r.layers.includes("he_affine");
      $("pinout").textContent = "click to drop a pin (H&E native pixel lookup)";
      requestAnimationFrame(() => { fit(); loadMini(); });
    },
  };
})();
