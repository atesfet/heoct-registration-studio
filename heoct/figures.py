"""Full-resolution figures (stage 06). Fig. 1 = pipeline panels a-g (+ Fig. 2 views h-j) at native input resolution.

Panels (each saved losslessly at the native resolution of its source):
  a  raw H&E                                  H&E native pixels (whole scan)
  b  H&E background-removed + Macenko         H&E native pixels (whole scan)
  c  raw OCT (dB, display-scaled)             OCT native grid (2 um/px)
  d  OCT background-removed + flat-field      OCT native grid (2 um/px)
  e  registered blend 50/50                   H&E-native grid over the OCT frame
  f  checkerboard A (500 um tiles)            H&E-native grid over the OCT frame
  g  checkerboard B (complement of A)         H&E-native grid over the OCT frame
  h  OCT (fixed) on the registered grid       H&E-native grid over the OCT frame
  i  H&E registered (stain-normalised)        H&E-native grid over the OCT frame
  j  false colour: OCT magenta, H&E OD green  H&E-native grid over the OCT frame
The composite places the panels in two rows (a-e, f-j) at a common height (the registered-grid height);
panels whose native height differs are rescaled for the composite only.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import tifffile

from . import io, qc

PANELS = [("a", "HE_raw"), ("b", "HE_bgremoved_macenko"), ("c", "OCT_raw"), ("d", "OCT_bgremoved_flatfield"),
          ("e", "registered_blend"), ("f", "registered_checkerboard_A_500um"), ("g", "registered_checkerboard_B_500um"),
          ("h", "OCT_registered_grid"), ("i", "HE_registered"), ("j", "registered_falsecolor")]
TITLES = {"a": "H&E raw", "b": "H&E bg-removed + Macenko", "c": "OCT raw (dB)", "d": "OCT bg-removed + flat-field",
          "e": "registered blend", "f": "checkerboard A", "g": "checkerboard B",
          "h": "OCT (fixed)", "i": "H&E registered", "j": "OCT magenta / H&E green"}


def _oct_display(db: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    v = np.nan_to_num(db, nan=0.0)
    ref = v[mask] if mask is not None and mask.any() else v[np.isfinite(db)]
    lo, hi = np.percentile(ref, [1, 99.5])
    return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def _he_od(rgb: np.ndarray) -> np.ndarray:
    """Summed optical density of stain-normalised H&E scaled to ~[0,1] (as used for registration)."""
    od = -np.log(np.clip(rgb.astype(np.float32), 1, None) / 255.0).sum(-1)
    return np.clip(od / max(float(np.percentile(od, 99.5)), 1e-6), 0, 1).astype(np.float32)


def _rgb(a: np.ndarray) -> np.ndarray:
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    return np.repeat(a[..., None], 3, -1) if a.ndim == 2 else a


def figure1(cfg: dict) -> Path:
    root = Path(cfg["output_root"]) / cfg["sample_id"]
    sid = cfg["sample_id"]
    out = root / "06_figures" / "fig1_fullres"; out.mkdir(parents=True, exist_ok=True)
    log = io.get_logger(out, "06_figures")
    d1, d2, d3, od = root / "01_he_preproc", root / "02_oct_preproc", root / "03_registration", root / "output"
    meta = json.loads((od / f"{sid}_metadata.json").read_text())
    he_px = meta["grids"]["he_native_grid"]["pixel_size_um"]
    oct_px = meta["grids"]["oct_grid"]["pixel_size_um"]

    oct_raw, _ = io.load_oct(cfg["oct_path"])
    oct_mask = tifffile.imread(d3 / "oct_tissue_mask_refined.tif") > 0
    oct_norm = tifffile.imread(d2 / "oct_norm.tif") * oct_mask
    he_reg = tifffile.imread(od / f"{sid}_HE_registered_stainnorm_he_native_grid.tif")
    db_reg = tifffile.imread(od / f"{sid}_OCT_dB_he_native_grid.tif")
    m_reg = tifffile.imread(od / f"{sid}_mask_OCT_he_native_grid.tif") > 0
    o_reg = _oct_display(db_reg, m_reg) * m_reg
    tile = int(round(500 / he_px))
    panels = {
        "a": io.load_he(cfg["he_path"]),
        "b": tifffile.imread(d1 / "he_stainnorm_rgb.tif"),
        "c": _rgb(_oct_display(oct_raw)),
        "d": _rgb(oct_norm.astype(np.float32)),
        "e": qc.blend(he_reg, o_reg),
        "f": qc.checkerboard(he_reg, o_reg, tile),
        "g": qc.checkerboard(he_reg, o_reg, tile, invert=True),
        "h": _rgb(o_reg),
        "i": he_reg,
        "j": qc.false_color(_he_od(he_reg), o_reg),
    }
    px = {**{k: f"{he_px} um" for k in "hij"}, "a": "H&E native (uncalibrated; ~%.3f um est.)" % he_px, "b": "H&E native (~%.3f um est.)" % he_px,
          "c": f"{oct_px} um", "d": f"{oct_px} um", "e": f"{he_px} um", "f": f"{he_px} um", "g": f"{he_px} um"}
    files = {}
    for key, name in PANELS:
        p = out / f"{sid}_fig1{key}_{name}.png"
        cv2.imwrite(str(p), panels[key][..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
        files[p.name] = {"panel": key, "title": TITLES[key], "shape": list(panels[key].shape), "pixel_size": px[key]}
        log.info(f"panel {key} {name} {panels[key].shape}")

    # composite: two rows (a-e, f-j) at the registered-grid height
    H = he_reg.shape[0]; gap = int(0.01 * H); band = int(0.05 * H)
    fs = band / 40.0

    def tile_panel(key):
        im = panels[key]; sc = H / im.shape[0]
        im = cv2.resize(im, (int(round(im.shape[1] * sc)), H), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LANCZOS4)
        head = np.full((band, im.shape[1], 3), 255, np.uint8)
        cv2.putText(head, key, (int(0.02 * im.shape[1]), int(0.75 * band)), cv2.FONT_HERSHEY_DUPLEX, fs * 1.3, (0, 0, 0), max(2, int(fs * 2.5)))
        cv2.putText(head, TITLES[key], (int(0.02 * im.shape[1] + band), int(0.72 * band)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.9, (0, 0, 0), max(1, int(fs * 1.6)))
        return np.vstack([head, im])

    def scalebar(im, x0, w):
        L = int(round(2000 / he_px)); y = band + int(0.96 * H); x1 = x0 + w - int(0.04 * w)
        cv2.rectangle(im, (x1 - L, y - max(4, H // 300)), (x1, y), (0, 0, 0), -1)
        cv2.putText(im, "2 mm", (x1 - L, y - H // 120), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (0, 0, 0), max(1, band // 25))

    rows = []
    for keys in (list("abcde"), list("fghij")):
        parts, xs, x = [], {}, 0
        for k in keys:
            t = tile_panel(k); xs[k] = (x, t.shape[1]); parts += [t, np.full((H + band, gap, 3), 255, np.uint8)]; x += t.shape[1] + gap
        row = np.hstack(parts[:-1])
        for k in ("e", "i"):
            if k in xs:
                scalebar(row, *xs[k])
        rows.append(row)
    W = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, W - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    comp = np.vstack([rows[0], np.full((gap, W, 3), 255, np.uint8), rows[1]])
    cp = out / f"{sid}_fig1_composite_fullres.png"
    cv2.imwrite(str(cp), comp[..., ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    files[cp.name] = {"panel": "composite", "shape": list(comp.shape),
                      "note": "all panels at common height = registered-grid height; individual panel files are native resolution"}
    io.save_png(out / f"{sid}_fig1_composite_preview.png", comp, 4000)
    io.write_json(out / "fig1_info.json", {"sample_id": sid, "description": __doc__, "files": files})
    # collected folder across samples (symlinks, no duplication)
    coll = Path(cfg["output_root"]) / "figures" / "fig1_fullres"; coll.mkdir(parents=True, exist_ok=True)
    for f in (cp, out / f"{sid}_fig1_composite_preview.png"):
        link = coll / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(f)
    log.info(f"composite {comp.shape} -> {cp}")
    return out


def run(cfg: dict) -> None:
    figure1(cfg)
