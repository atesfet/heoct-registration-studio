#!/usr/bin/env python
"""Visual check that the epidermis survives the OCT, registered-H&E and common tissue masks.

For each sample: an overview and 6 zoomed edge views (800 um) evenly spaced around the tissue boundary.
Panels: OCT + mask outlines | registered H&E + outlines | OCT inside the common mask | H&E inside the common mask.
Outlines: cyan = common mask, magenta = OCT mask, green = registered H&E mask.
Output: <output_root>/figures/epidermis_check/<sample>_epidermis_check.png

Usage: python scripts/epidermis_check.py [--root <results root>] [sample ...]
       (default root: output_root in configs/default.yaml; default samples: every registered sample under the root)
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, tifffile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heoct import io

COL = {"common": (0, 255, 255), "oct": (255, 0, 255), "he": (0, 210, 0)}   # RGB


def outline(img, masks, th):
    out = img.copy()
    for k in ("oct", "he", "common"):
        cs, _ = cv2.findContours(masks[k].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, COL[k], th)
    return out


def label(w, text, h=26):
    b = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(b, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return b


def run(sample, root):
    o = root / sample / "output"
    db = tifffile.imread(o / f"{sample}_OCT_dB_oct_grid.tif")
    he = tifffile.imread(o / f"{sample}_HE_registered_stainnorm_oct_grid.tif")
    M = {"common": tifffile.imread(o / f"{sample}_mask_common_oct_grid.tif") > 0,
         "oct": tifffile.imread(o / f"{sample}_mask_OCT_oct_grid.tif") > 0,
         "he": tifffile.imread(o / f"{sample}_mask_HE_registered_oct_grid.tif") > 0}
    meta = json.loads((o / f"{sample}_metadata.json").read_text())
    edge_med = ((meta["accuracy"].get("tissue_edge_distance") or {}).get("deformable") or {}).get("median_um")
    v = np.nan_to_num(db, nan=0.0); lo, hi = np.percentile(v[M["oct"]], [1, 99.5])
    oc = np.repeat(((np.clip((v - lo) / (hi - lo), 0, 1)) * 255).astype(np.uint8)[..., None], 3, -1)
    oct_in = np.where(M["common"][..., None], oc, 0).astype(np.uint8)
    he_in = np.where(M["common"][..., None], he, 255).astype(np.uint8)
    H, W = M["common"].shape
    # overview (OCT and H&E with outlines), ~900 px tall
    f = 900 / H
    ov = [cv2.resize(outline(x, M, max(3, int(3 / f))), None, fx=f, fy=f, interpolation=cv2.INTER_AREA) for x in (oc, he)]
    # 6 edge points evenly around the common-mask boundary (by angle from the centroid), 400 px = 800 um crops
    ys, xs = np.nonzero(M["common"]); cy, cx = ys.mean(), xs.mean()
    e = M["common"] & ~(cv2.erode(M["common"].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    ey, ex = np.nonzero(e); ang = np.degrees(np.arctan2(-(ey - cy), ex - cx)) % 360
    r = 200; rows = []
    for k, a0 in enumerate(range(0, 360, 60)):
        sel = np.abs(((ang - a0 + 180) % 360) - 180) < 5
        if not sel.any():
            continue
        d = np.hypot(ey[sel] - cy, ex[sel] - cx); i = np.flatnonzero(sel)[np.argmax(d)]   # outermost point in that direction
        y0 = int(np.clip(ey[i] - r, 0, H - 2 * r)); x0 = int(np.clip(ex[i] - r, 0, W - 2 * r)); sl = np.s_[y0:y0 + 2 * r, x0:x0 + 2 * r]
        Ms = {kk: m[sl] for kk, m in M.items()}
        panels = [outline(oc[sl], Ms, 2), outline(he[sl], Ms, 2), oct_in[sl], he_in[sl]]
        cv2.line(panels[0], (2 * r - 110, 2 * r - 12), (2 * r - 10, 2 * r - 12), (255, 255, 255), 3)   # 100 px = 200 um
        cv2.putText(panels[0], "200 um", (2 * r - 112, 2 * r - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        gap = np.full((2 * r, 6, 3), 255, np.uint8)
        row = np.hstack([panels[0], gap, panels[1], gap, panels[2], gap, panels[3]])
        rows.append(np.vstack([label(row.shape[1], f"edge {a0:3d} deg  (x {x0 * 2 + 400} um, y {y0 * 2 + 400} um)   |  OCT + outlines   |   H&E + outlines   |   OCT inside common mask   |   H&E inside common mask"), row]))
    Wd = rows[0].shape[1]
    top = np.hstack(ov); top = np.pad(top, ((0, 0), (0, max(0, Wd - top.shape[1])), (0, 0)), constant_values=255)[:, :Wd]
    head = label(Wd, f"{sample}: epidermis check   outlines: cyan = COMMON mask, magenta = OCT mask, green = registered H&E mask   |   OCT/H&E tissue-edge distance (median) = {edge_med:.1f} um", 30)
    fig = np.vstack([head, top] + rows)
    out = root / "figures" / "epidermis_check" / f"{sample}_epidermis_check.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), fig[..., ::-1])
    return out


if __name__ == "__main__":
    import yaml
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("samples", nargs="*"); a = ap.parse_args()
    root = Path(a.root or yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text())["output_root"])
    samples = a.samples or sorted(d.name for d in root.iterdir() if d.is_dir() and any((d / "output").glob("*_metadata.json")))
    for s in samples:
        print(run(s, root))
