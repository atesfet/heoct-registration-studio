"""OCT en-face preprocessing: robust background removal and tile flat-field correction."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from .he_preproc import _disk, keep_large_components


def tile_flatfield(img: np.ndarray, mask: np.ndarray, tile: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove the additive (dB) intensity pattern shared by all acquisition tiles.

    The en-face mosaic is built from tile x tile patches; each patch carries the same
    vignetting / focus fall-off. We estimate it as the robust median over tiles of
    (patch - patch median) on tissue pixels and subtract it.
    """
    h, w = img.shape
    ny, nx = h // tile, w // tile
    stack = []
    for iy in range(ny):
        for ix in range(nx):
            sl = np.s_[iy * tile:(iy + 1) * tile, ix * tile:(ix + 1) * tile]
            m = mask[sl]
            if m.mean() < 0.9:
                continue
            p = img[sl].copy()
            p[~m] = np.nan
            stack.append(p - np.nanmedian(p))
    if len(stack) < 4:
        return img, np.zeros((tile, tile), np.float32)
    pattern = np.nanmedian(np.stack(stack), axis=0)
    pattern = cv2.GaussianBlur(np.nan_to_num(pattern).astype(np.float32), (0, 0), 4)
    pattern -= np.median(pattern)
    full = np.tile(pattern, (ny + 1, nx + 1))[:h, :w]
    return img - full, pattern


def tissue_mask(oct_img: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """Robust OCT tissue mask.

    Tissue is detected by texture rather than raw intensity: background (noise floor, empty tiles,
    mounting medium) contains a high local fraction of near-noise pixels, tissue does not.
    1) local dark fraction  fr = G_sigma * [value < dark_db or NaN]
    2) candidate = finite & fr < max_dark_frac, small opening, largest component(s), fill holes
    3) separate debris / neighbouring tissue pieces attached by bridges narrower than ~2*body_open_um
       (opening -> largest body -> dilation back -> intersect)
    Morphology runs at 1/work_downsample resolution for speed and is upsampled at the end.
    """
    finite = np.isfinite(oct_img)
    v = np.nan_to_num(oct_img, nan=0.0)
    k = int(cfg.get("work_downsample", 4))
    px = float(cfg.get("px_um", 2.0)) * k
    dark = ((v < cfg["dark_db"]) | ~finite).astype(np.float32)
    small = lambda a: cv2.resize(a, (a.shape[1] // k, a.shape[0] // k), interpolation=cv2.INTER_AREA)
    fr = cv2.GaussianBlur(small(dark), (0, 0), cfg["dark_sigma_um"] / px)
    fin = small(finite.astype(np.float32)) > 0.5
    fg = (fr < cfg["max_dark_frac"]) & fin
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _disk(max(1, int(round(cfg["open_um"] / px))))).astype(bool)
    fg = keep_large_components(fg, cfg["min_component_frac"])
    fg = ndi.binary_fill_holes(fg)
    R = max(1, int(round(cfg["body_open_um"] / px)))
    opened = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, _disk(R)).astype(bool)
    lab, n = ndi.label(opened)
    if n:
        areas = ndi.sum(opened, lab, np.arange(1, n + 1))
        keep = np.flatnonzero(areas >= cfg["min_component_frac"] * areas.max()) + 1
        # every foreground pixel goes to its nearest opened component (Voronoi split), so neighbouring
        # pieces that survive the opening cannot be re-attached to the main body
        dist, (iy, ix) = ndi.distance_transform_edt(~opened, return_indices=True)
        near = lab[iy, ix]
        fg = fg & np.isin(near, keep) & (dist <= R + 2)
    fg = keep_large_components(fg, cfg["min_component_frac"])
    if cfg.get("split_touching", True):
        fg, split_info = split_touching_pieces(fg, cfg.get("neck_ratio", 0.6))
    else:
        split_info = {}
    m = ndi.binary_fill_holes(fg)
    full = cv2.resize(m.astype(np.uint8) * 255, (oct_img.shape[1], oct_img.shape[0]), interpolation=cv2.INTER_LINEAR) > 127
    full &= finite | ndi.binary_fill_holes(finite)   # never extend beyond the reconstructed footprint
    return full, {"split": split_info, "work_downsample": k, "dark_db": cfg["dark_db"], "max_dark_frac": cfg["max_dark_frac"],
                  "body_open_um": cfg["body_open_um"]}


def split_touching_pieces(fg: np.ndarray, neck_ratio: float) -> tuple[np.ndarray, dict]:
    """Detach neighbouring tissue pieces that touch the main section through a neck.

    Watershed on the distance transform splits the mask into convex-ish lobes. A lobe is dropped only
    if the neck joining it to the main lobe (max inscribed radius along the shared border) is narrower than
    neck_ratio x the smaller of the two lobes' inscribed radii -> irregular single sections stay intact.
    """
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed
    dt = ndi.distance_transform_edt(fg)
    if dt.max() < 5:
        return fg, {}
    markers, n = ndi.label(h_maxima(dt, max(3.0, 0.04 * dt.max())))
    if n < 2:
        return fg, {"n_lobes": int(n)}
    ws = watershed(-dt, markers, mask=fg)
    areas = ndi.sum(fg, ws, np.arange(1, n + 1))
    main = int(np.argmax(areas)) + 1
    keep = {main}
    info = {"n_lobes": int(n), "dropped": []}
    rad = ndi.maximum(dt, ws, np.arange(1, n + 1))
    changed = True
    while changed:                                   # grow the kept set through wide necks
        changed = False
        kept = np.isin(ws, list(keep))
        border = cv2.dilate(kept.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~kept & fg
        for l in set(np.unique(ws[border]).tolist()) - keep - {0}:
            neck = dt[border & (ws == l)].max()
            if neck >= neck_ratio * min(rad[l - 1], rad[main - 1]):
                keep.add(l); changed = True
    for l in set(range(1, n + 1)) - keep:
        info["dropped"].append({"lobe": l, "area_px": float(areas[l - 1]), "radius_px": float(rad[l - 1])})
    return np.isin(ws, list(keep)), info


def run(oct_img: np.ndarray, cfg: dict) -> dict:
    finite = np.isfinite(oct_img)
    mask, minfo = tissue_mask(oct_img, cfg)
    img = np.where(finite, oct_img, 0.0).astype(np.float32)
    pattern = np.zeros((1, 1), np.float32)
    if cfg["flatfield"]:
        img, pattern = tile_flatfield(img, mask & finite, cfg["tile_size_px"])
    lo, hi = np.percentile(img[mask & finite], cfg["clip_percentiles"])
    norm = np.clip((img - lo) / (hi - lo), 0, 1)
    norm[~mask] = 0
    clean = img.copy()
    clean[~mask] = np.nan
    info = {"mask": minfo, "clip_db": [float(lo), float(hi)], "mask_fraction": float(mask.mean()),
            "finite_fraction": float(finite.mean()),
            "removed_debris_fraction": float((finite & ~mask).mean()),
            "flatfield_pattern_range_db": [float(pattern.min()), float(pattern.max())]}
    return {"mask": mask, "clean_db": clean, "norm": norm.astype(np.float32), "flatfield": pattern, "info": info}
