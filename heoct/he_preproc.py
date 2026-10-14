"""H&E preprocessing: tissue/background segmentation and Macenko stain normalisation."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi


def _disk(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def keep_large_components(mask: np.ndarray, min_frac: float) -> np.ndarray:
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    areas = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.flatnonzero(areas >= min_frac * areas.max()) + 1
    return np.isin(lab, keep)


def estimate_background_rgb(rgb_small: np.ndarray) -> np.ndarray:
    """Per-channel brightness of the empty glass (bright, unsaturated pixels)."""
    hsv = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2HSV)
    v, s = hsv[..., 2].astype(float), hsv[..., 1].astype(float) / 255
    cand = (s < 0.08) & (v > np.percentile(v, 50))
    bg = np.median(rgb_small[cand].astype(float), axis=0)
    return np.clip(bg, 200, 255)


def tissue_mask(rgb: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (full-res filled tissue mask, full-res raw stained-pixel mask, info)."""
    ds = cfg["work_downsample"]
    small = cv2.resize(rgb, (rgb.shape[1] // ds, rgb.shape[0] // ds), interpolation=cv2.INTER_AREA)
    bg = estimate_background_rgb(small)
    od = (-np.log(np.clip(small.astype(float), 1, None) / bg[None, None, :])).clip(0)
    od_sum = od.sum(-1)
    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    hue, sat = hsv[..., 0].astype(int), hsv[..., 1] / 255.0
    m = (od_sum > cfg["min_od"]) & (sat > cfg["min_saturation"])
    for lo, hi in cfg["exclude_hue_ranges"]:
        m &= ~((hue >= lo) & (hue <= hi))
    raw = m.copy()
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, _disk(1)).astype(bool)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, _disk(cfg["close_radius_px"])).astype(bool)
    m = keep_large_components(m, cfg["min_component_frac"])
    m = ndi.binary_fill_holes(m)
    # small pale gaps (clefts) touching the boundary are not filled; that is intended.
    full = cv2.resize(m.astype(np.uint8) * 255, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR) > 127
    raw_full = cv2.resize(raw.astype(np.uint8) * 255, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
    info = {"background_rgb": bg, "tissue_fraction": float(full.mean()), "work_downsample": ds}
    return full, raw_full & full, info


# ---------------------------------------------------------------- Macenko
def _od(rgb: np.ndarray, io: np.ndarray) -> np.ndarray:
    return (-np.log(np.clip(rgb.astype(np.float32), 1, None) / io[None, :].astype(np.float32))).clip(0)


def macenko_fit(rgb_pixels: np.ndarray, io: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Fit stain matrix (3x2, columns H,E) and 99th-pct concentrations from tissue pixels (N,3)."""
    od = _od(rgb_pixels, io)
    od = od[np.all(od > beta, axis=1)]
    _, vecs = np.linalg.eigh(np.cov(od.T))
    vecs = vecs[:, [1, 2]]
    if vecs[0, 0] < 0: vecs[:, 0] *= -1
    if vecs[0, 1] < 0: vecs[:, 1] *= -1
    proj = od @ vecs
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    a_min, a_max = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = vecs @ np.array([np.cos(a_min), np.sin(a_min)])
    v2 = vecs @ np.array([np.cos(a_max), np.sin(a_max)])
    he = np.stack([v1, v2], 1) if v1[0] > v2[0] else np.stack([v2, v1], 1)  # H has larger red OD
    he /= np.linalg.norm(he, axis=0, keepdims=True)
    conc = np.linalg.lstsq(he, od.T, rcond=None)[0]
    max_c = np.percentile(conc, 99, axis=1)
    return he, max_c


def macenko_apply(rgb: np.ndarray, mask: np.ndarray, io: np.ndarray, he: np.ndarray, max_c: np.ndarray,
                  ref_he: np.ndarray, ref_max_c: np.ndarray, chunk: int = 4_000_000):
    """Normalise rgb; returns (normalised RGB uint8 with white background, H conc, E conc)."""
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    mflat = mask.reshape(-1)
    out = np.full_like(flat, 255)
    ch = np.zeros((h * w, 2), np.float32)
    pinv = np.linalg.pinv(he).astype(np.float32)
    scale = (ref_max_c / max_c).astype(np.float32)
    ref_he = ref_he.astype(np.float32)
    for s in range(0, flat.shape[0], chunk):
        sl = slice(s, s + chunk)
        idx = np.flatnonzero(mflat[sl]) + s
        if idx.size == 0:
            continue
        c = (_od(flat[idx], io) @ pinv.T).clip(0)
        ch[idx] = c
        c = c * scale[None, :]
        out[idx] = np.clip(255.0 * np.exp(-(c @ ref_he.T)), 0, 255).astype(np.uint8)
    return out.reshape(h, w, 3), ch[:, 0].reshape(h, w), ch[:, 1].reshape(h, w)


def run(rgb: np.ndarray, cfg: dict, rng_seed: int = 0) -> dict:
    mask, raw, info = tissue_mask(rgb, cfg)
    io = info["background_rgb"]
    rng = np.random.default_rng(rng_seed)
    idx = np.flatnonzero(raw.reshape(-1))
    idx = rng.choice(idx, size=min(idx.size, 1_000_000), replace=False)
    sn = cfg["stain_norm"]
    he, max_c = macenko_fit(rgb.reshape(-1, 3)[idx], io, sn["alpha"], sn["beta"])
    norm, hc, ec = macenko_apply(rgb, mask, io, he, max_c, np.array(sn["ref_he"]), np.array(sn["ref_max_c"]))
    bgrem = rgb.copy()
    bgrem[~mask] = 255
    info |= {"stain_matrix_he_columns": he, "max_concentration": max_c,
             "ref_he": sn["ref_he"], "ref_max_c": sn["ref_max_c"]}
    return {"mask": mask, "bg_removed": bgrem, "stain_norm": norm, "hema": hc, "eosin": ec, "info": info}
