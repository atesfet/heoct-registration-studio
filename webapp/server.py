"""H&E <-> OCT Registration Studio: local web server (standard library http.server, no framework).

Endpoints (JSON unless noted)
  GET  /                         index.html
  GET  /static/<file>            front-end assets
  GET  /files/<abs path>         a file inside a registered-results folder (QC pages, PNGs, TIFFs, PDF); C:/... on Windows
  GET  /api/system               GPU / environment / defaults
  GET  /api/browse?path=&mode=   folder listing for the file browser (dirs + .tif/.tiff/.yaml/.yml/.json)
  POST /api/detect   {folder}    find the H&E (*4x*.tif) and OCT (*topk*.tif) of a sample folder
  POST /api/scan     {folder}    list sample sub-folders of a data root (for batch runs)
  POST /api/inspect  {he_path, oct_path}  shapes, metadata and thumbnails
  POST /api/run      {jobs, output_root, overrides, stages}   start the pipeline (subprocess)
  GET  /api/job?since=N          job state, progress, new log lines
  POST /api/cancel               stop the running job
  GET  /api/results?root=        registered samples under an output root (+ metrics)
  GET  /api/result?dir=          one registered sample: metadata summary, QC images, ROIs, viewer info
  GET  /api/tile?dir=&layer=&x=&y=&w=&h=&ow=&oh=   image/png crop of a layer on the OCT grid (viewer)
  GET  /api/coord?dir=&x=&y=     OCT pixel -> um, scanner mm and H&E native pixel
  POST /api/open     {path}      open a folder in the desktop file manager (local use)
"""
from __future__ import annotations

import base64
import json
import os
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from collections import OrderedDict
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import tifffile
import yaml

APP = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(APP))
from heoct import io as hio  # noqa: E402

VERSION = "Registration Studio " + (APP / "VERSION").read_text().strip()
CONF = {"token": None, "allow_roots": [], "local_only": True}


# ------------------------------------------------------------------ helpers
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o); return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, bytes):
        return base64.b64encode(o).decode()
    return o


def allowed(p: Path) -> bool:
    if not CONF["allow_roots"]:
        return True
    rp = p.resolve()
    return any(rp == r or r in rp.parents for r in CONF["allow_roots"])


def check_allowed(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not allowed(p):
        raise PermissionError(f"{p} is outside the allowed roots")
    return p


def png_b64(img: np.ndarray, max_side: int = 900) -> str:
    if max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if img.ndim == 3:
        img = img[..., ::-1]
    return base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def oct_u8(db: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    v = np.nan_to_num(db, nan=0.0)
    ref = v[mask] if mask is not None and mask.any() else v[np.isfinite(db)]
    lo, hi = np.percentile(ref, [1, 99.5]) if ref.size else (0, 1)
    return (np.clip((v - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------ system
def system_info():
    info = {"version": VERSION, "python": sys.version.split()[0], "app": str(APP), "home": str(Path.home()),
            "sep": os.sep, "allow_roots": [str(r) for r in CONF["allow_roots"]], "local_only": CONF["local_only"]}
    try:
        import torch
        from heoct.device import DEVICE, device_name
        info["torch"] = torch.__version__
        info["device"] = DEVICE
        info["gpu"] = device_name() if DEVICE in ("cuda", "mps") else None
        if DEVICE == "cuda":
            free, total = torch.cuda.mem_get_info()
            info["gpu_mem"] = f"{free / 2**30:.0f}/{total / 2**30:.0f} GB free"
    except Exception as e:  # noqa: BLE001
        info["torch"] = None; info["gpu"] = None; info["device"] = None; info["torch_error"] = str(e)
    info["cpus"] = os.cpu_count()
    info["tectonic"] = bool(shutil.which("tectonic") or (Path(sys.executable).parent / "tectonic").exists())
    info["defaults"] = yaml.safe_load((APP / "configs" / "default.yaml").read_text())
    return info


# ------------------------------------------------------------------ file browser
SHOW_EXT = (".tif", ".tiff", ".yaml", ".yml", ".json")


def find_pair(folder: Path):
    tifs = [p for p in folder.glob("*") if p.suffix.lower() in (".tif", ".tiff") and p.is_file()]
    oct_ = sorted([p for p in tifs if "topk" in p.name.lower() or "recon_xy" in p.name.lower()])
    he = sorted([p for p in tifs if p not in oct_ and re.search(r"(4x|he|h&e|hne)", p.name.lower())])
    return (he[0] if he else None), (oct_[0] if oct_ else None)


def is_registered(folder: Path) -> bool:
    return any((folder / "output").glob("*_metadata.json"))


def browse(path: str | None, mode: str = "dir"):
    p = Path(path).expanduser() if path else (CONF["allow_roots"][0] if CONF["allow_roots"] else Path.home())
    if not p.exists():
        p = p.parent if p.parent.exists() else Path.home()
    if p.is_file():
        p = p.parent
    if not allowed(p):
        p = CONF["allow_roots"][0]
    entries = []
    try:
        for c in sorted(p.iterdir(), key=lambda c: c.name.lower()):
            if c.name.startswith("."):
                continue
            try:
                if c.is_dir():
                    he, oc = (find_pair(c) if os.access(c, os.R_OK) else (None, None))
                    entries.append({"name": c.name, "path": str(c), "type": "dir",
                                    "is_pair": bool(he and oc), "is_registered": is_registered(c),
                                    "is_results_root": c.name == "data_12um_registered" or any(is_registered(x) for x in list(c.iterdir())[:40] if x.is_dir())})
                elif c.suffix.lower() in SHOW_EXT:
                    st = c.stat()
                    kind = ("OCT" if ("topk" in c.name.lower() or "recon_xy" in c.name.lower()) else
                            "H&E" if re.search(r"(4x|he|h&e)", c.name.lower()) and c.suffix.lower() in (".tif", ".tiff") else "")
                    entries.append({"name": c.name, "path": str(c), "type": "file", "size_mb": round(st.st_size / 2**20, 1),
                                    "kind": kind})
            except OSError:
                continue
            if len(entries) > 2000:
                break
    except PermissionError:
        pass
    if CONF["allow_roots"]:
        roots = [str(r) for r in CONF["allow_roots"]]
    elif os.name == "nt":
        import string
        roots = [f"{d}:\\" for d in string.ascii_uppercase if Path(f"{d}:\\").exists()]
    else:
        roots = ["/"] + [str(x) for x in (Path("/media"), Path("/mnt"), Path("/Volumes")) if x.exists()]
    he, oc = find_pair(p)
    parent = str(p.parent) if p.parent != p and allowed(p.parent) else None
    return {"path": str(p), "parent": parent, "entries": entries, "home": str(Path.home()), "roots": roots,
            "is_pair": bool(he and oc), "is_registered": is_registered(p)}


def detect(folder: str):
    f = check_allowed(folder)
    if f.is_file():
        f = f.parent
    he, oc = find_pair(f)
    return {"folder": str(f), "sample_id": f.name, "he_path": str(he) if he else None, "oct_path": str(oc) if oc else None}


def scan(folder: str):
    root = check_allowed(folder)
    out = []
    for c in sorted(root.iterdir(), key=lambda c: c.name.lower()):
        if c.is_dir() and not c.name.startswith("."):
            he, oc = find_pair(c)
            if he and oc:
                out.append({"sample_id": c.name, "folder": str(c), "he_path": str(he), "oct_path": str(oc)})
    return {"root": str(root), "samples": out}


def inspect(he_path: str, oct_path: str):
    res = {}
    if he_path:
        hp = check_allowed(he_path)
        with tifffile.TiffFile(hp) as t:
            s = t.series[0]
            res["he"] = {"path": str(hp), "shape": list(s.shape), "dtype": str(s.dtype), "size_mb": round(hp.stat().st_size / 2**20, 1),
                         "software": (t.pages[0].tags["Software"].value if "Software" in t.pages[0].tags else None)}
            lv = t.series[-1].asarray() if len(t.series) > 1 and max(t.series[-1].shape[:2]) >= 120 else None
        im = hio.load_he(str(hp)) if lv is None or max(lv.shape[:2]) < 400 else lv[..., :3]
        res["he"]["thumb"] = png_b64(im)
    if oct_path:
        op = check_allowed(oct_path)
        o, meta = hio.load_oct(str(op))
        res["oct"] = {"path": str(op), "shape": list(o.shape), "dtype": str(o.dtype), "size_mb": round(op.stat().st_size / 2**20, 1),
                      "px_um": meta.get("px_um"), "mode": meta.get("oct_mode"), "unit": meta.get("oct_value_unit"),
                      "nan_frac": float(np.isnan(o).mean()), "extent_mm": [o.shape[1] * meta.get("px_um", 2) / 1000, o.shape[0] * meta.get("px_um", 2) / 1000],
                      "thumb": png_b64(oct_u8(o))}
    return res


# ------------------------------------------------------------------ job (pipeline subprocess)
STAGE_NAMES = {"01": "H&E preprocessing", "02": "OCT preprocessing", "03": "registration", "04": "QC",
               "05": "export (output/)", "06": "full-resolution figures"}


class Job:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"; self.logs: list[str] = []; self.progress = {}; self.error = None
        self.proc = None; self.cancel = threading.Event(); self.started = None; self.finished = None
        self.request = None; self.results = []

    def log(self, line: str):
        with self.lock:
            self.logs.append(line.rstrip("\n"))

    def snapshot(self, since: int):
        with self.lock:
            return {"state": self.state, "progress": dict(self.progress), "logs": self.logs[since:], "n_logs": len(self.logs),
                    "error": self.error, "started": self.started, "finished": self.finished,
                    "results": list(self.results), "request": self.request}

    def start(self, req: dict):
        if self.state in ("starting", "running"):
            raise RuntimeError("a job is already running")
        jobs = req.get("jobs") or []
        if not jobs:
            raise ValueError("no samples to run")
        out_root = check_allowed(req["output_root"]); out_root.mkdir(parents=True, exist_ok=True)
        for j in jobs:
            check_allowed(j["he_path"]); check_allowed(j["oct_path"])
            if not re.fullmatch(r"[A-Za-z0-9._-]+", j["sample_id"] or ""):
                raise ValueError(f"invalid sample name {j['sample_id']!r} (letters, digits, . _ - only)")
        self.cancel.clear()
        with self.lock:
            self.state = "starting"; self.logs = []; self.error = None; self.results = []
            self.started = time.time(); self.finished = None; self.request = req
            self.progress = {"sample_i": 0, "n_samples": len(jobs), "stage": None, "frac": 0.0}
        threading.Thread(target=self._run, args=(req, out_root), daemon=True).start()

    def _run(self, req, out_root):
        stages = req.get("stages") or ["01", "02", "03", "04", "05"]
        defaults = yaml.safe_load((APP / "configs" / "default.yaml").read_text())
        overrides = req.get("overrides") or {}
        try:
            if (req.get("yaml_overrides") or "").strip():
                extra = yaml.safe_load(req["yaml_overrides"]) or {}
                if not isinstance(extra, dict):
                    raise ValueError("YAML overrides must be a mapping (key: value)")
                overrides = hio._merge(overrides, extra)
                self.log("YAML overrides applied: " + json.dumps(extra))
            self.state = "running"
            for si, j in enumerate(req["jobs"]):
                if self.cancel.is_set():
                    break
                sid = j["sample_id"]
                cfg = hio._merge(defaults, overrides)
                cfg |= {"sample_id": sid, "he_path": j["he_path"], "oct_path": j["oct_path"], "output_root": str(out_root)}
                sdir = out_root / sid; sdir.mkdir(parents=True, exist_ok=True)
                cpath = sdir / "run_config.yaml"
                cpath.write_text("# written by Registration Studio " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n" + yaml.safe_dump(cfg, sort_keys=False))
                self.log(f"===== [{si + 1}/{len(req['jobs'])}] {sid}  config -> {cpath}")
                with self.lock:
                    self.progress.update({"sample_i": si, "sample": sid, "stage": None, "frac": si / len(req["jobs"])})
                cmd = [sys.executable, "-u", str(APP / "scripts" / "run_pipeline.py"), "--config", str(cpath),
                       "--default", str(APP / "configs" / "default.yaml"), "--stages", *stages]
                self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                             cwd=str(APP), env={**os.environ, "PYTHONUNBUFFERED": "1"}, start_new_session=True)
                sub_n = 0
                for line in self.proc.stdout:
                    if "Warning" in line and "warn(" in line:
                        continue
                    self.log(line)
                    m = re.search(r"\[(\d\d)_", line)
                    if m and m.group(1) in stages:
                        st = m.group(1)
                        if self.progress.get("stage") != st:
                            sub_n = 0
                        sub_n += ("level" in line or "candidate" in line)
                        k = stages.index(st)
                        # stage 03 has ~17 progress lines (6 candidates, 3+3 affine, 5 deform)
                        inner = min(sub_n / 17.0, 0.95) if st == "03" else 0.5
                        w = [1, 1, 8, 1, 1, 1]  # rough relative cost per stage
                        ws = [w[["01", "02", "03", "04", "05", "06"].index(x)] for x in stages]
                        f = (sum(ws[:k]) + inner * ws[k]) / sum(ws)
                        with self.lock:
                            self.progress.update({"stage": st, "stage_name": STAGE_NAMES.get(st, st),
                                                  "frac": (si + f) / len(req["jobs"])})
                    if self.cancel.is_set():
                        break
                rc = self.proc.wait()
                if self.cancel.is_set():
                    self.log("cancelled"); break
                if rc != 0:
                    raise RuntimeError(f"{sid}: pipeline exited with code {rc} (see log)")
                with self.lock:
                    self.results.append(str(sdir))
            with self.lock:
                self.state = "cancelled" if self.cancel.is_set() else "done"
                self.progress["frac"] = 1.0 if self.state == "done" else self.progress.get("frac", 0)
        except Exception as e:  # noqa: BLE001
            with self.lock:
                self.state = "error"; self.error = f"{type(e).__name__}: {e}"
            self.log("ERROR " + self.error)
        finally:
            self.finished = time.time(); self.proc = None

    def stop(self):
        self.cancel.set()
        p = self.proc
        if p and p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:  # noqa: BLE001
                p.terminate()


JOB = Job()


# ------------------------------------------------------------------ results
def _sample_summary(d: Path) -> dict | None:
    mj = next((d / "output").glob("*_metadata.json"), None)
    if not mj:
        return None
    m = json.loads(mj.read_text())
    acc = m.get("accuracy", {})
    lt = (acc.get("landmark_tre") or {}).get("deformable") or {}
    la = (acc.get("landmark_tre") or {}).get("affine") or {}
    rd = (acc.get("residual_block_matching") or {}).get("deformable") or {}
    init = (m.get("transform") or {}).get("init") or {}
    return {"sample_id": m.get("sample_id"), "dir": str(d), "created": m.get("created"),
            "he_px_um": m["inputs"]["he"].get("pixel_size_um_estimated"), "oct_px_um": m["inputs"]["oct"].get("pixel_size_um"),
            "mirrored": (init.get("best_flip", 1) < 0), "rotation_deg": init.get("best_rot_deg"),
            "tre_aff_med": la.get("median_um"), "tre_med": lt.get("median_um"), "tre_p90": lt.get("p90_um"), "tre_n": lt.get("n_matched"),
            "tre_lt2": lt.get("frac_below_2um"), "resid_med": rd.get("median_um"),
            "dice": (acc.get("mask_dice") or {}).get("deformable"),
            "edge_med": ((acc.get("tissue_edge_distance") or {}).get("deformable") or {}).get("median_um"),
            "oct_shape": m["grids"]["oct_grid"]["shape_rows_cols"], "he_path": m["inputs"]["he"]["path"], "oct_path": m["inputs"]["oct"]["path"]}


def results(root: str):
    r = check_allowed(root)
    out = []
    cands = [r] + sorted([c for c in r.iterdir() if c.is_dir()], key=lambda c: c.name.lower()) if r.is_dir() else []
    for c in cands:
        s = _sample_summary(c)
        if s:
            out.append(s)
    aux = Path((yaml.safe_load((APP / "configs" / "default.yaml").read_text()) or {}).get("aux_root") or r)
    rep = next((d / "report" / "report.pdf" for d in (r, aux) if (d / "report" / "report.pdf").exists()), None)
    figs = next((d / "figures" for d in (r, aux) if (d / "figures").exists()), None)
    return {"root": str(r), "samples": out, "report_pdf": str(rep) if rep else None,
            "figures": sorted(str(p) for p in figs.rglob("*_preview.png")) if figs else []}


def result(dirpath: str):
    d = check_allowed(dirpath)
    s = _sample_summary(d)
    if not s:
        raise FileNotFoundError(f"{d} has no output/*_metadata.json (not a registered sample)")
    qc = d / "04_qc"
    rep = json.loads((qc / "qc_report.json").read_text()) if (qc / "qc_report.json").exists() else {}
    gallery = []
    for sub, label in (("overview", "overview"), ("residuals", "accuracy"), ("rois", "zoomed ROIs")):
        for p in sorted((qc / sub).glob("*.png")):
            gallery.append({"group": label, "name": p.stem, "path": str(p)})
    for p in sorted((d / "06_figures").rglob("*_preview.png")):
        gallery.insert(0, {"group": "full-resolution figure", "name": p.stem, "path": str(p)})
    for p in sorted((d / "01_he_preproc" / "preview").glob("*.png")) + sorted((d / "02_oct_preproc" / "preview").glob("*.png")):
        gallery.append({"group": "preprocessing", "name": p.stem, "path": str(p)})
    outputs = []
    for p in sorted((d / "output").glob("*")):
        if p.is_file():
            outputs.append({"name": p.name, "path": str(p), "size_mb": round(p.stat().st_size / 2**20, 1)})
    layers = VIEW.layers_available(d)
    return {"summary": s, "rois": rep.get("rois", []), "orientation": rep.get("orientation"), "deformation": rep.get("deformation"),
            "landmark": {k: rep.get(f"landmark_{k}") for k in ("affine", "deformable")},
            "residual": {k: rep.get(f"residual_{k}") for k in ("affine", "deformable")},
            "gallery": gallery, "outputs": outputs, "layers": layers,
            "qc_html": str(qc / "index.html") if (qc / "index.html").exists() else None}


# ------------------------------------------------------------------ viewer: layer pyramids + tiles
class Viewer:
    """Loads a registered sample's OCT-grid layers once, keeps 2x pyramids, and serves crops."""

    def __init__(self, max_samples=3):
        self.cache: OrderedDict[str, dict] = OrderedDict(); self.max = max_samples; self.lock = threading.Lock()

    @staticmethod
    def _files(d: Path, sid: str) -> dict:
        o = d / "output"
        return {"he": o / f"{sid}_HE_registered_stainnorm_oct_grid.tif",
                "he_orig": o / f"{sid}_HE_registered_original_oct_grid.tif",
                "he_affine": d / "03_registration" / "he_stainnorm_warped_affine.tif",
                "oct": o / f"{sid}_OCT_dB_oct_grid.tif",
                "mask_common": o / f"{sid}_mask_common_oct_grid.tif",
                "mask_oct": o / f"{sid}_mask_OCT_oct_grid.tif",
                "mask_he": o / f"{sid}_mask_HE_registered_oct_grid.tif",
                "he_mask_affine": d / "03_registration" / "he_mask_warped_affine.tif"}

    def layers_available(self, d: Path) -> list[str]:
        s = _sample_summary(d)
        if not s:
            return []
        f = self._files(d, s["sample_id"])
        L = [k for k in ("he", "he_orig", "he_affine", "oct") if f[k].exists()]
        if f["he"].exists():
            L += ["he_od"]
        if f["he_affine"].exists():
            L += ["he_od_affine"]
        if f["mask_oct"].exists():
            L += ["contour", "contour_affine"] if f["he_mask_affine"].exists() else ["contour"]
        return L

    def get(self, d: Path) -> dict:
        key = str(d.resolve())
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key); return self.cache[key]
        s = _sample_summary(d)
        f = self._files(d, s["sample_id"])
        L = {}
        mo = tifffile.imread(f["mask_oct"]) > 0 if f["mask_oct"].exists() else None
        for k in ("he", "he_orig", "he_affine"):
            if f[k].exists():
                L[k] = tifffile.imread(f[k])
        if f["oct"].exists():
            db = tifffile.imread(f["oct"]); L["oct"] = oct_u8(db, mo)
        for k, src in (("he_od", "he"), ("he_od_affine", "he_affine")):
            if src in L:
                od = -np.log(np.clip(L[src].astype(np.float32), 1, None) / 255.0).sum(-1)
                L[k] = (np.clip(od / max(np.percentile(od, 99.5), 1e-6), 0, 1) * 255).astype(np.uint8)
        for k, mh in (("contour", f["mask_he"]), ("contour_affine", f["he_mask_affine"])):
            if mo is not None and mh.exists():
                c = np.zeros((*mo.shape, 4), np.uint8)
                for m, col in ((mo, (255, 0, 255, 255)), (tifffile.imread(mh) > 0, (0, 230, 0, 255))):
                    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(c, cs, -1, col, 3)
                L[k] = c
        pyr = {}
        for k, a in L.items():
            levels = [a]
            while max(levels[-1].shape[:2]) > 512:
                b = levels[-1]
                levels.append(cv2.resize(b, (max(1, b.shape[1] // 2), max(1, b.shape[0] // 2)),
                                         interpolation=cv2.INTER_NEAREST if k.startswith("contour") else cv2.INTER_AREA))
            pyr[k] = levels
        mp = d / "03_registration" / "map_oct_px_to_he_px.npy"
        meta = json.loads(next((d / "output").glob("*_metadata.json")).read_text())
        entry = {"pyr": pyr, "shape": L["oct"].shape[:2] if "oct" in L else None,
                 "map": np.load(mp, mmap_mode="r") if mp.exists() else None, "meta": meta}
        with self.lock:
            self.cache[key] = entry
            while len(self.cache) > self.max:
                self.cache.popitem(last=False)
        return entry

    def tile(self, d: Path, layer: str, x: float, y: float, w: float, h: float, ow: int, oh: int) -> bytes:
        e = self.get(d)
        if layer not in e["pyr"]:
            raise KeyError(f"layer {layer!r} not available")
        ow, oh = int(np.clip(ow, 1, 4096)), int(np.clip(oh, 1, 4096))
        scale = ow / max(w, 1e-6)                    # output px per OCT px
        lvl = 0
        while lvl + 1 < len(e["pyr"][layer]) and scale <= 0.5 ** (lvl + 1):
            lvl += 1
        a = e["pyr"][layer][lvl]; f = 2 ** lvl
        H0, W0 = e["pyr"][layer][0].shape[:2]
        fx, fy = a.shape[1] / W0, a.shape[0] / H0
        # affine map from output pixel -> level pixel (pixel centres)
        sx, sy = (w * fx) / ow, (h * fy) / oh
        M = np.array([[sx, 0, x * fx + 0.5 * sx - 0.5], [0, sy, y * fy + 0.5 * sy - 0.5]], np.float32)
        nearest = scale / (1 / f) >= 1.5 or layer.startswith("contour")
        bv = (0, 0, 0, 0) if layer.startswith("contour") else (255, 255, 255) if layer.startswith("he") and not layer.startswith("he_od") else 0
        out = cv2.warpAffine(a, M, (ow, oh), flags=(cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR) | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=bv)
        if out.ndim == 3 and out.shape[2] == 3:
            out = out[..., ::-1]
        elif out.ndim == 3 and out.shape[2] == 4:
            out = out[..., [2, 1, 0, 3]]
        return cv2.imencode(".png", out, [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()

    def coord(self, d: Path, x: float, y: float) -> dict:
        e = self.get(d); m = e["meta"]
        px = m["grids"]["oct_grid"]["pixel_size_um"]
        x0, y0 = m["inputs"]["oct"]["x0_mm"], m["inputs"]["oct"]["y0_mm"]
        res = {"oct_px": [x, y], "um": [(x + 0.5) * px, (y + 0.5) * px],
               "scanner_mm": [x0 + x * px / 1000, y0 + y * px / 1000]}
        if e["map"] is not None:
            H, W = e["map"].shape[1:]
            xi, yi = int(np.clip(round(x), 0, W - 1)), int(np.clip(round(y), 0, H - 1))
            res["he_px"] = [float(e["map"][0, yi, xi]), float(e["map"][1, yi, xi])]
        return res


VIEW = Viewer()


def open_in_file_manager(path: str):
    if not CONF["local_only"]:
        raise PermissionError("opening folders is only available when the server runs locally")
    p = check_allowed(path)
    p = p if p.is_dir() else p.parent
    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", str(p)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    else:
        os.startfile(str(p))  # type: ignore[attr-defined]
    return {"ok": True}


# ------------------------------------------------------------------ HTTP
MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml", ".json": "application/json",
        ".pdf": "application/pdf", ".txt": "text/plain; charset=utf-8", ".log": "text/plain; charset=utf-8",
        ".yaml": "text/plain; charset=utf-8", ".tif": "image/tiff", ".npz": "application/octet-stream"}


class Handler(BaseHTTPRequestHandler):
    server_version = "RegistrationStudio/1"

    def log_message(self, fmt, *args):  # quieter console
        if "/api/tile" in (args[0] if args else ""):
            return
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- auth (optional shared token; cookie set after first ?token= visit)
    def _authed(self, q) -> bool:
        if not CONF["token"]:
            return True
        c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
        if "regstudio" in c and secrets.compare_digest(c["regstudio"].value, CONF["token"]):
            return True
        t = (q.get("token") or [""])[0]
        if t and secrets.compare_digest(t, CONF["token"]):
            self._set_cookie = True
            return True
        return False

    def _send(self, code, body, ctype="application/json"):
        if ctype == "application/json":
            body = json.dumps(_jsonable(body)).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if getattr(self, "_set_cookie", False):
            self.send_header("Set-Cookie", f"regstudio={CONF['token']}; Path=/; HttpOnly; SameSite=Strict")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, p: Path):
        if not p.is_file():
            return self._send(404, {"error": f"not found: {p}"})
        self._send(200, p.read_bytes(), MIME.get(p.suffix.lower(), "application/octet-stream"))

    def do_GET(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        g = lambda k, d=None: (q.get(k) or [d])[0]  # noqa: E731
        if not self._authed(q):
            return self._send(401, b"<h3>Registration Studio: access token required</h3><p>Open the link with <code>?token=...</code>.</p>", "text/html")
        try:
            if u.path in ("/", "/index.html"):
                return self._file(STATIC / "index.html")
            if u.path.startswith("/static/"):
                p = (STATIC / u.path[len("/static/"):]).resolve()
                if STATIC not in p.parents:
                    return self._send(403, {"error": "forbidden"})
                return self._file(p)
            if u.path.startswith("/files/"):
                # /files/<absolute path with forward slashes>; Windows drives as /files/C:/...
                rel = urllib.parse.unquote(u.path[len("/files/"):])
                p = Path(rel if re.match(r"^[A-Za-z]:", rel) else "/" + rel).resolve()
                # only files that belong to a registered sample (or its results root) can be served
                # registered-sample folders, or the cross-sample report/ and figures/ folders (aux_root)
                ok = allowed(p) and (any(is_registered(a) for a in list(p.parents)[:6])
                                     or any(a.name in ("report", "figures") for a in list(p.parents)[:3]))
                if not ok:
                    return self._send(403, {"error": "only files inside registered-result folders can be served"})
                return self._file(p)
            if u.path == "/api/system":
                return self._send(200, system_info())
            if u.path == "/api/browse":
                return self._send(200, browse(g("path"), g("mode", "dir")))
            if u.path == "/api/job":
                return self._send(200, JOB.snapshot(int(g("since", 0))))
            if u.path == "/api/results":
                return self._send(200, results(g("root")))
            if u.path == "/api/result":
                return self._send(200, result(g("dir")))
            if u.path == "/api/tile":
                d = check_allowed(g("dir"))
                b = VIEW.tile(d, g("layer"), float(g("x")), float(g("y")), float(g("w")), float(g("h")), int(g("ow")), int(g("oh")))
                return self._send(200, b, "image/png")
            if u.path == "/api/coord":
                return self._send(200, VIEW.coord(check_allowed(g("dir")), float(g("x")), float(g("y"))))
            return self._send(404, {"error": f"unknown path {u.path}"})
        except Exception as e:  # noqa: BLE001
            return self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if not self._authed(q):
            return self._send(401, {"error": "access token required"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/detect":
                return self._send(200, detect(body["folder"]))
            if u.path == "/api/scan":
                return self._send(200, scan(body["folder"]))
            if u.path == "/api/inspect":
                return self._send(200, inspect(body.get("he_path"), body.get("oct_path")))
            if u.path == "/api/run":
                JOB.start(body); return self._send(200, {"ok": True})
            if u.path == "/api/cancel":
                JOB.stop(); return self._send(200, {"ok": True})
            if u.path == "/api/open":
                return self._send(200, open_in_file_manager(body["path"]))
            return self._send(404, {"error": f"unknown path {u.path}"})
        except Exception as e:  # noqa: BLE001
            return self._send(400, {"error": f"{type(e).__name__}: {e}"})


def serve(host="127.0.0.1", port=8780, open_browser=True, token=None, allow_roots=()):
    CONF["token"] = token
    CONF["allow_roots"] = [Path(r).expanduser().resolve() for r in allow_roots]
    CONF["local_only"] = host in ("127.0.0.1", "localhost", "::1")
    for p in range(port, port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                port = p; break
    srv = ThreadingHTTPServer((host, port), Handler)
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    url = f"http://{shown}:{port}/" + (f"?token={token}" if token else "")
    print(f"Registration Studio running at {url}", flush=True)
    if not CONF["local_only"] and not token:
        print("WARNING: listening on a non-local interface without --token; anyone on the network can use it.", flush=True)
    if open_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        JOB.stop(); srv.server_close()
