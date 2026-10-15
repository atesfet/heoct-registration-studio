"use strict";
const $ = (id) => document.getElementById(id);
const S = { sys: null, logN: 0, poll: null, sep: "/", tab: "single", batch: [], results: null };

// ------------------------------------------------------------------ utils
async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function toast(msg, err = false) {
  const t = $("toast"); t.textContent = msg; t.className = "toast" + (err ? " err" : "");
  clearTimeout(t._h); t._h = setTimeout(() => t.classList.add("hidden"), err ? 8000 : 3500);
}
function busy(btn, on) { btn.disabled = on; btn.classList.toggle("spin", on); }
function joinPath(a, b) { if (!a) return b; return a.replace(/[\\/]+$/, "") + S.sep + b; }
function esc(s) { return String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function f2(v, d = 2) { return v === null || v === undefined ? "–" : Number(v).toFixed(d); }
// path-style URL so relative links inside served HTML (QC pages) keep working; Windows "C:\\x" -> /files/C:/x
function fileUrl(p) { return "/files/" + p.replace(/\\/g, "/").replace(/^\/+/, "").split("/").map(encodeURIComponent).join("/"); }
function baseName(p) { return (p || "").split(/[\\/]/).filter(Boolean).pop() || ""; }

const SAVE_KEYS = ["sample_folder", "he_path", "oct_path", "data_root", "output_root", "sample_id", "results_root"];
function save() { try { const st = {}; for (const k of SAVE_KEYS) st[k] = $(k).value; localStorage.setItem("regstudio", JSON.stringify(st)); } catch (e) {} }
function restore() { try { const st = JSON.parse(localStorage.getItem("regstudio") || "{}"); for (const k of SAVE_KEYS) if (st[k]) $(k).value = st[k]; } catch (e) {} }

// ------------------------------------------------------------------ system
async function loadSystem() {
  const s = await api("/api/system"); S.sys = s; S.sep = s.sep || "/";
  $("version").textContent = s.version;
  const b = [];
  b.push(s.gpu ? `<span class="badge ok">GPU · ${esc(s.gpu)}${s.gpu_mem ? " · " + esc(s.gpu_mem) : ""}</span>` : `<span class="badge warn">no GPU · CPU (slow)</span>`);
  b.push(`<span class="badge">torch ${esc(s.torch || "missing")}</span>`);
  b.push(`<span class="badge">${s.cpus} CPU threads</span>`);
  if (!s.local_only) b.push(`<span class="badge warn">network mode</span>`);
  if (s.allow_roots.length) b.push(`<span class="badge">roots: ${esc(s.allow_roots.join(", "))}</span>`);
  $("sysbadges").innerHTML = b.join("");
  $("default-cfg").textContent = toYaml(s.defaults);
  const R = s.defaults.registration || {}, O = s.defaults.oct || {};
  $("rot_step_deg").value = R.rot_step_deg ?? 10; $("allow_flip").value = String(R.allow_flip ?? true);
  if (R.he_px_um_search) { $("px_lo").value = R.he_px_um_search[0]; $("px_hi").value = R.he_px_um_search[1]; }
  $("save_fullres").checked = !!(s.defaults.figures || {}).save_fullres_previews;
  $("max_dark_frac").value = O.max_dark_frac ?? 0.35; $("body_open_um").value = O.body_open_um ?? 240;
  if (!$("output_root").value && s.defaults.output_root) $("output_root").value = s.defaults.output_root;
  if (!$("results_root").value) $("results_root").value = $("output_root").value;
  updatePath();
}
function toYaml(o, ind = "") {
  if (o === null || typeof o !== "object") return JSON.stringify(o);
  if (Array.isArray(o)) return "[" + o.map((v) => (typeof v === "object" ? JSON.stringify(v) : JSON.stringify(v))).join(", ") + "]";
  return Object.entries(o).map(([k, v]) => (v && typeof v === "object" && !Array.isArray(v) ? `${ind}${k}:\n${toYaml(v, ind + "  ")}` : `${ind}${k}: ${toYaml(v)}`)).join("\n");
}

// ------------------------------------------------------------------ input
function setTab(t) {
  S.tab = t;
  document.querySelectorAll("#input-tabs .tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === t));
  $("tab-single").classList.toggle("hidden", t !== "single"); $("tab-batch").classList.toggle("hidden", t !== "batch");
  $("sample_id").disabled = t === "batch"; updatePath();
}
async function detect() {
  const f = $("sample_folder").value.trim(); if (!f) return toast("Choose a sample folder first", true);
  try {
    const d = await api("/api/detect", { folder: f });
    if (d.he_path) $("he_path").value = d.he_path; if (d.oct_path) $("oct_path").value = d.oct_path;
    $("sample_id").value = d.sample_id; updatePath(); save();
    if (!d.he_path || !d.oct_path) toast(`Found ${d.he_path ? "H&E" : "no H&E"} and ${d.oct_path ? "OCT" : "no OCT"} in ${d.folder}`, true);
    else inspect();
  } catch (e) { toast(e.message, true); }
}
async function inspect() {
  const he = $("he_path").value.trim(), oc = $("oct_path").value.trim();
  if (!he && !oc) return toast("Choose the H&E and OCT images", true);
  const btn = $("btn-inspect"); busy(btn, true); $("inspect-status").textContent = "reading…";
  try {
    const r = await api("/api/inspect", { he_path: he, oct_path: oc });
    const kv = [];
    if (r.he) { kv.push(["H&E size", `${r.he.shape[1]} × ${r.he.shape[0]}`], ["H&E file", `${r.he.size_mb} MB · ${r.he.shape[2] || 1} ch`]); $("he-thumb").src = "data:image/png;base64," + r.he.thumb; }
    if (r.oct) {
      kv.push(["OCT size", `${r.oct.shape[1]} × ${r.oct.shape[0]}`], ["OCT pixel", `${r.oct.px_um} µm`], ["OCT field", `${r.oct.extent_mm[0].toFixed(1)} × ${r.oct.extent_mm[1].toFixed(1)} mm`],
              ["OCT background (NaN)", `${(100 * r.oct.nan_frac).toFixed(0)} %`], ["OCT values", r.oct.unit || "–"], ["OCT mode", r.oct.mode || "–"]);
      $("oct-thumb").src = "data:image/png;base64," + r.oct.thumb;
    }
    $("scan-summary").innerHTML = kv.map(([k, v]) => `<div class="kv"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join("");
    $("inspect-out").classList.remove("hidden"); $("inspect-status").textContent = "";
    if (!$("sample_id").value && he) $("sample_id").value = baseName(he.replace(/[\\/][^\\/]+$/, "")); updatePath();
  } catch (e) { toast(e.message, true); $("inspect-status").textContent = ""; }
  busy(btn, false);
}
async function scanRoot() {
  const f = $("data_root").value.trim(); if (!f) return toast("Choose a data root", true);
  try {
    const r = await api("/api/scan", { folder: f }); S.batch = r.samples;
    $("batch-table").querySelector("tbody").innerHTML = r.samples.length ? r.samples.map((s, i) =>
      `<tr><td><input type="checkbox" class="bsel" data-i="${i}" checked></td><td><b>${esc(s.sample_id)}</b></td><td>${esc(baseName(s.he_path))}</td><td>${esc(baseName(s.oct_path))}</td></tr>`).join("")
      : `<tr><td colspan="4" class="muted">no sub-folder with both an H&amp;E (*4x*.tif) and an OCT (*topk*.tif) image</td></tr>`;
    toast(`${r.samples.length} sample(s) found`); save();
  } catch (e) { toast(e.message, true); }
}
function updatePath() {
  const root = $("output_root").value.trim(), id = S.tab === "batch" ? "<sample>" : $("sample_id").value.trim() || "<sample>";
  $("final-path").textContent = root ? joinPath(root, id) : "…";
}

// ------------------------------------------------------------------ parameters
const PRESETS = {
  fast: { deform_levels_um: [16, 8, 4, 2], deform_ctrl_spacing_um: [800, 400, 200, 100], deform_iters: [300, 250, 200, 150], deform_lambda: [1.0, 0.5, 0.25, 0.1] },
  standard: {},
  accurate: { deform_levels_um: [16, 8, 4, 2, 2, 2], deform_ctrl_spacing_um: [800, 400, 200, 100, 50, 25], deform_iters: [450, 400, 300, 250, 250, 200], deform_lambda: [1.0, 0.5, 0.25, 0.1, 0.05, 0.05] },
  affine: { deform_levels_um: [], deform_ctrl_spacing_um: [], deform_iters: [], deform_lambda: [] },
};
function num(id) { const v = Number($(id).value); if (!Number.isFinite(v)) throw new Error(`${id}: not a number`); return v; }
function buildOverrides() {
  return {
    registration: { ...PRESETS[$("preset").value], allow_flip: $("allow_flip").value === "true", rot_step_deg: num("rot_step_deg"),
                    he_px_um_search: [num("px_lo"), num("px_hi")] },
    oct: { max_dark_frac: num("max_dark_frac"), body_open_um: num("body_open_um") },
    figures: { save_fullres_previews: $("save_fullres").checked },
  };
}
function stages() {
  const st = [...document.querySelectorAll(".stage:checked")].map((c) => c.value);
  return $("save_fullres").checked ? [...st, "06"] : st;
}
function formState() {
  const st = {}; for (const id of ["preset", "allow_flip", "rot_step_deg", "px_lo", "px_hi", "max_dark_frac", "body_open_um", "yaml_overrides", ...SAVE_KEYS]) st[id] = $(id).value;
  st.stages = [...document.querySelectorAll(".stage:checked")].map((c) => c.value); st.save_fullres = $("save_fullres").checked; return st;
}
function applyState(st) {
  for (const [k, v] of Object.entries(st)) if ($(k) && typeof v !== "object") $(k).value = v;
  if (st.stages) document.querySelectorAll(".stage").forEach((c) => (c.checked = st.stages.includes(c.value)));
  $("save_fullres").checked = !!st.save_fullres;
  updatePath(); save();
}

// ------------------------------------------------------------------ run
function buildJobs() {
  if (S.tab === "batch") {
    const sel = [...document.querySelectorAll(".bsel:checked")].map((c) => S.batch[+c.dataset.i]);
    if (!sel.length) throw new Error("select at least one sample (Scan the data root first)");
    return sel.map((s) => ({ sample_id: s.sample_id, he_path: s.he_path, oct_path: s.oct_path }));
  }
  const he = $("he_path").value.trim(), oc = $("oct_path").value.trim(), id = $("sample_id").value.trim();
  if (!he || !oc) throw new Error("choose the H&E and OCT images");
  if (!id) throw new Error("enter a sample name");
  return [{ sample_id: id, he_path: he, oct_path: oc }];
}
async function run() {
  let req;
  try {
    const out = $("output_root").value.trim(); if (!out) throw new Error("choose an output root");
    req = { jobs: buildJobs(), output_root: out, overrides: buildOverrides(), yaml_overrides: $("yaml_overrides").value, stages: stages() };
    if (!req.stages.length) throw new Error("select at least one stage");
  } catch (e) { return toast(e.message, true); }
  try {
    await api("/api/run", req);
    S.logN = 0; $("log").textContent = ""; $("result").innerHTML = "";
    $("progress-wrap").classList.remove("hidden"); $("btn-run").disabled = true; $("btn-cancel").classList.remove("hidden");
    startPolling(); save();
  } catch (e) { toast(e.message, true); }
}
function startPolling() { clearInterval(S.poll); S.poll = setInterval(pollJob, 1000); pollJob(); }
async function pollJob() {
  let j; try { j = await api("/api/job?since=" + S.logN); } catch (e) { return; }
  if (j.logs.length) { const L = $("log"); L.textContent += j.logs.join("\n") + "\n"; L.scrollTop = L.scrollHeight; S.logN = j.n_logs; }
  const p = j.progress || {}; $("progress-bar").style.width = `${Math.round(100 * (p.frac || 0))}%`;
  const el = j.started ? Math.round(((j.finished || Date.now() / 1000) - j.started)) : 0;
  $("progress-text").textContent = `${j.state}` + (p.sample ? ` · sample ${p.sample_i + 1}/${p.n_samples} ${p.sample}` : "") +
    (p.stage ? ` · stage ${p.stage} ${p.stage_name || ""}` : "") + ` · ${Math.floor(el / 60)} min ${el % 60} s`;
  if (["done", "error", "cancelled"].includes(j.state)) { clearInterval(S.poll); $("btn-run").disabled = false; $("btn-cancel").classList.add("hidden"); showResult(j); }
}
function showResult(j) {
  const r = $("result");
  if (j.state === "done") {
    r.innerHTML = `<div class="note ok"><b>Done.</b> ${j.results.map((d) => `<code>${esc(d)}</code>`).join(" ")}
      <div class="row" style="margin-top:6px"><button class="secondary sm" id="btn-view-last">Open in viewer</button></div></div>`;
    $("btn-view-last").onclick = async () => { $("results_root").value = j.request.output_root; await loadResults(); openResult(j.results[j.results.length - 1]); };
    $("results_root").value = j.request.output_root; loadResults();
  } else if (j.state === "error") r.innerHTML = `<div class="note warn"><b>Error:</b> ${esc(j.error)}</div>`;
  else r.innerHTML = `<div class="note warn">Cancelled.</div>`;
}

// ------------------------------------------------------------------ results
async function loadResults() {
  const root = $("results_root").value.trim(); if (!root) return toast("Choose a results folder", true);
  try {
    const r = await api("/api/results?root=" + encodeURIComponent(root)); S.results = r; save();
    const tb = $("results-table").querySelector("tbody");
    tb.innerHTML = r.samples.length ? r.samples.map((s, i) => `<tr data-dir="${esc(s.dir)}">
      <td><b>${esc(s.sample_id)}</b></td><td>${s.mirrored ? "yes" : "no"}</td><td class="num">${f2(s.rotation_deg, 1)}</td><td class="num">${f2(s.he_px_um, 3)}</td>
      <td class="num">${f2(s.tre_aff_med)}</td><td class="num ${s.tre_med <= 3 ? "good" : "meh"}">${f2(s.tre_med)}</td><td class="num">${f2(s.tre_p90, 1)}</td>
      <td class="num">${s.tre_lt2 == null ? "–" : Math.round(100 * s.tre_lt2) + "%"}</td><td class="num">${f2(s.resid_med)}</td><td class="num">${f2(s.dice, 4)}</td>
      <td><button class="sm" data-view="${i}">View</button></td></tr>`).join("")
      : `<tr><td colspan="11" class="muted">no registered samples (folders with output/*_metadata.json) under this folder</td></tr>`;
    tb.querySelectorAll("[data-view]").forEach((b) => (b.onclick = () => openResult(r.samples[+b.dataset.view].dir)));
    const rl = $("report-link"); rl.classList.toggle("hidden", !r.report_pdf); if (r.report_pdf) rl.href = fileUrl(r.report_pdf);
  } catch (e) { toast(e.message, true); }
}
async function openResult(dir) {
  try {
    const r = await api("/api/result?dir=" + encodeURIComponent(dir));
    document.querySelectorAll("#results-table tbody tr").forEach((tr) => tr.classList.toggle("sel", tr.dataset.dir === dir));
    const s = r.summary;
    $("viewer-wrap").classList.remove("hidden");
    $("viewer-title").textContent = `${s.sample_id} — ${s.oct_shape[1]} × ${s.oct_shape[0]} OCT px @ ${s.oct_px_um} µm`;
    $("qc-link").href = r.qc_html ? fileUrl(r.qc_html) : "#";
    $("btn-open-folder").onclick = () => api("/api/open", { path: s.dir }).catch((e) => toast(e.message, true));
    const lt = r.landmark.deformable || {}, la = r.landmark.affine || {};
    const kv = [["Landmark TRE (deformable)", `${f2(lt.median_um)} µm median · p90 ${f2(lt.p90_um, 1)}`], ["Landmark TRE (affine)", `${f2(la.median_um)} µm`],
                ["Landmarks < 2 µm", lt.frac_below_2um == null ? "–" : `${Math.round(100 * lt.frac_below_2um)} % (n=${lt.n_matched})`],
                ["Patch residual", `${f2((r.residual.deformable || {}).median_um)} µm median`], ["Tissue Dice", f2(s.dice, 4)],
                ["Orientation", `${s.mirrored ? "mirrored, " : ""}${f2(s.rotation_deg, 1)}°`], ["H&E pixel (est.)", `${f2(s.he_px_um, 3)} µm`],
                ["Non-rigid displacement", r.deformation ? `${f2(r.deformation.median_um, 1)} µm median · max ${f2(r.deformation.max_um, 0)}` : "–"]];
    $("viewer-metrics").innerHTML = kv.map(([k, v]) => `<div class="kv"><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join("");
    // gallery
    const groups = [...new Set(r.gallery.map((g) => g.group))];
    const filt = $("gal-filter"); filt.innerHTML = ["all", ...groups].map((g, i) => `<button data-g="${esc(g)}" class="${i ? "" : "on"}">${esc(g)}</button>`).join("");
    const draw = (g) => { $("gallery").innerHTML = r.gallery.filter((x) => g === "all" || x.group === g).map((x) =>
      `<figure data-src="${esc(fileUrl(x.path))}" data-name="${esc(x.name)}"><img loading="lazy" src="${esc(fileUrl(x.path))}"><figcaption title="${esc(x.name)}">${esc(x.name)}</figcaption></figure>`).join("");
      $("gallery").querySelectorAll("figure").forEach((f) => (f.onclick = () => lightbox(f.dataset.src, f.dataset.name))); };
    filt.querySelectorAll("button").forEach((b) => (b.onclick = () => { filt.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b)); draw(b.dataset.g); }));
    draw("all");
    $("outputs").innerHTML = r.outputs.map((o) => `<li><a href="${esc(fileUrl(o.path))}" target="_blank">${esc(o.name)}</a> <span>${o.size_mb} MB</span></li>`).join("");
    window.RegViewer.open(r);
    $("viewer-wrap").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (e) { toast(e.message, true); }
}
function lightbox(src, name) { $("lb-img").src = src; $("lb-title").textContent = name; $("lb-open").href = src; $("lightbox").classList.remove("hidden"); }

// ------------------------------------------------------------------ file / folder browser (as in oct_recon_AF_v2)
let B = { target: null, mode: "dir", path: null };
function openBrowser(target, mode) {
  let start = $(target).value.trim();
  if (!start && (target === "he_path" || target === "oct_path")) start = $("sample_folder").value.trim();
  if (!start && target === "results_root") start = $("output_root").value.trim();
  B = { target, mode, path: start };
  $("modal-title").textContent = mode === "dir" ? "Select folder" : (target === "he_path" ? "Select H&E image" : target === "oct_path" ? "Select OCT image" : "Select file");
  $("modal-select").classList.toggle("hidden", mode !== "dir");
  $("modal").classList.remove("hidden"); loadDir(B.path);
}
async function loadDir(path) {
  try {
    const d = await api("/api/browse?path=" + encodeURIComponent(path || "") + "&mode=" + B.mode);
    B.path = d.path; $("modal-path").value = d.path; B.parent = d.parent; B.home = d.home;
    const roots = $("modal-roots"); roots.innerHTML = '<option value="">Go to…</option>' + d.roots.map((r) => `<option>${esc(r)}</option>`).join("");
    const ul = $("modal-list"); ul.innerHTML = "";
    const want = B.target === "he_path" ? "H&E" : B.target === "oct_path" ? "OCT" : null;
    for (const e of d.entries) {
      if (B.mode === "dir" && e.type !== "dir") continue;
      if (B.mode === "file" && e.type === "file" && !/\.tiff?$/i.test(e.name) && (B.target === "he_path" || B.target === "oct_path")) continue;
      const li = document.createElement("li");
      li.className = (e.is_pair || e.is_registered || (want && e.kind === want)) ? "vol" : "";
      const tags = [e.is_pair ? '<span class="tag">H&amp;E + OCT pair</span>' : "", e.is_registered ? '<span class="tag reg">registered</span>' : "",
                    e.type === "file" && e.kind ? `<span class="tag">${esc(e.kind)}</span>` : ""].join("");
      li.innerHTML = `${e.type === "dir" ? "📁" : "🖼️"} <span>${esc(e.name)}</span>${tags}${e.type === "file" ? `<span class="size">${e.size_mb} MB</span>` : ""}`;
      li.onclick = () => (e.type === "dir" ? loadDir(e.path) : pick(e.path));
      ul.appendChild(li);
    }
    $("modal-note").textContent = d.is_pair ? "✓ this folder contains an H&E + OCT pair" : d.is_registered ? "✓ this folder is a registered sample" : "";
  } catch (e) { toast(e.message, true); }
}
function pick(p) {
  $(B.target).value = p; $("modal").classList.add("hidden"); updatePath(); save();
  if (B.target === "sample_folder") detect();
  if (B.target === "data_root") scanRoot();
  if (B.target === "results_root") loadResults();
  if ((B.target === "he_path" || B.target === "oct_path") && $("he_path").value && $("oct_path").value) inspect();
}

// ------------------------------------------------------------------ wiring
window.addEventListener("DOMContentLoaded", async () => {
  restore();
  document.querySelectorAll("[data-browse]").forEach((b) => (b.onclick = () => openBrowser(b.dataset.browse, b.dataset.mode)));
  document.querySelectorAll("#input-tabs .tab").forEach((b) => (b.onclick = () => setTab(b.dataset.tab)));
  $("btn-detect").onclick = detect; $("btn-inspect").onclick = inspect; $("btn-scan").onclick = scanRoot;
  $("sample_folder").addEventListener("keydown", (e) => { if (e.key === "Enter") detect(); });
  $("data_root").addEventListener("keydown", (e) => { if (e.key === "Enter") scanRoot(); });
  $("results_root").addEventListener("keydown", (e) => { if (e.key === "Enter") loadResults(); });
  $("batch-all").onchange = (e) => document.querySelectorAll(".bsel").forEach((c) => (c.checked = e.target.checked));
  for (const id of ["output_root", "sample_id", "he_path", "oct_path"]) $(id).oninput = () => { updatePath(); save(); };
  $("btn-run").onclick = run; $("btn-cancel").onclick = () => api("/api/cancel", {}).then(() => toast("Cancelling…"));
  $("btn-results").onclick = loadResults;
  $("btn-save-cfg").onclick = () => { const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([JSON.stringify({ app: "registration-studio", state: formState(), overrides: buildOverrides() }, null, 2)], { type: "application/json" }));
    a.download = `${$("sample_id").value || "registration"}_studio_config.json`; a.click(); };
  $("load-cfg").onchange = async (e) => { const f = e.target.files[0]; if (!f) return;
    try { const t = await f.text(); if (/\.ya?ml$/i.test(f.name)) { $("yaml_overrides").value = t; document.querySelector("details").open = true; toast("YAML loaded into overrides"); }
          else { const c = JSON.parse(t); applyState(c.state || c); toast("Config loaded"); } } catch (err) { toast("Invalid config: " + err.message, true); } };
  $("modal-close").onclick = () => $("modal").classList.add("hidden");
  $("modal-up").onclick = () => B.parent && loadDir(B.parent); $("modal-home").onclick = () => loadDir(B.home);
  $("modal-roots").onchange = (e) => e.target.value && loadDir(e.target.value);
  $("modal-path").addEventListener("keydown", (e) => { if (e.key === "Enter") loadDir(e.target.value); });
  $("modal-select").onclick = () => pick(B.path);
  $("lb-close").onclick = () => $("lightbox").classList.add("hidden");
  $("lightbox").onclick = (e) => { if (e.target.id === "lightbox") $("lightbox").classList.add("hidden"); };
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") { $("modal").classList.add("hidden"); $("lightbox").classList.add("hidden"); } });
  try { await loadSystem(); } catch (e) { toast("Cannot reach the server: " + e.message, true); }
  try { const j = await api("/api/job?since=0"); if (["starting", "running"].includes(j.state)) { $("progress-wrap").classList.remove("hidden"); $("btn-run").disabled = true; $("btn-cancel").classList.remove("hidden"); startPolling(); } } catch (e) {}
  if ($("results_root").value) loadResults();
});
