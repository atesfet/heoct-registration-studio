"""File I/O, logging and small shared helpers."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import yaml


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, default: str | Path | None = None) -> dict:
    """Load a per-sample config deep-merged over configs/default.yaml."""
    path = Path(path).resolve()
    default = Path(default) if default else path.parents[1] / "default.yaml"
    cfg = yaml.safe_load(open(default)) if default.exists() else {}
    cfg = _merge(cfg, yaml.safe_load(open(path)))
    cfg["_config_path"] = str(path)
    cfg["_default_config_path"] = str(default.resolve()) if default.exists() else None
    return cfg


def stage_dir(cfg: dict, stage: str) -> Path:
    d = Path(cfg["output_root"]) / cfg["sample_id"] / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logger(out_dir: Path, name: str) -> logging.Logger:
    log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s [%(name)s] %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(), logging.FileHandler(out_dir / f"{name}.log", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def write_json(path: Path, obj: dict) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return str(o)
    obj = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), **obj}
    path.write_text(json.dumps(obj, indent=2, default=default))


def load_he(path: str) -> np.ndarray:
    """Return H&E as uint8 RGB (alpha dropped)."""
    im = tifffile.imread(path)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    return np.ascontiguousarray(im)


def load_oct(path: str) -> tuple[np.ndarray, dict]:
    """Return OCT en-face float32 image (NaN = no tissue) and metadata."""
    with tifffile.TiffFile(path) as t:
        im = t.asarray().astype(np.float32)
        meta = dict(t.imagej_metadata or {})
        try:
            meta["software_json"] = json.loads(t.pages[0].tags["Software"].value)
            meta["software_json"].get("metadata", {}).get("x", {}).pop("values", None)
            meta["software_json"].get("metadata", {}).get("y", {}).pop("values", None)
            meta["software_json"].get("metadata", {}).get("x", {}).pop("index", None)
            meta["software_json"].get("metadata", {}).get("y", {}).pop("index", None)
        except Exception:
            pass
    meta["px_um"] = float(meta.get("oct_pixel_x_um", 2.0))
    return im, meta


def save_tif(path: Path, arr: np.ndarray, px_um: float | None = None, desc: dict | None = None) -> None:
    kw = {}
    if px_um:
        kw["resolution"] = (1e4 / px_um, 1e4 / px_um)
        kw["resolutionunit"] = "CENTIMETER"
    md = {"unit": "um"} | ({"px_um": px_um} if px_um else {}) | (desc or {})
    tifffile.imwrite(path, arr, compression="zlib", metadata=None,
                     description=json.dumps(md, default=str), **kw)


def save_png(path: Path, arr: np.ndarray, max_side: int | None = None) -> None:
    """Save RGB (or gray) uint8 image as PNG, optionally downsampled."""
    if max_side and max(arr.shape[:2]) > max_side:
        s = max_side / max(arr.shape[:2])
        arr = cv2.resize(arr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if arr.ndim == 3:
        arr = arr[..., ::-1]
    cv2.imwrite(str(path), arr)


def to_uint8(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((x - lo) / (hi - lo + 1e-12), 0, 1) * 255).astype(np.uint8)
