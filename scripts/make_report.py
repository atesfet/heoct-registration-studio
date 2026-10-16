#!/usr/bin/env python
"""Build a Nature-Methods-style LaTeX report (and PDF) over all registered samples.

Usage: python scripts/make_report.py [--samples 12umA sample_10 ...]
Output: <aux_root>/report/{report.tex, report.pdf, figures/*.png}  (aux_root defaults to output_root)
"""
import argparse, json, shutil, subprocess, sys, time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from heoct import io, qc  # noqa: E402

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6})
COL = {"affine": "#9aa5b1", "deformable": "#1f6feb"}


def load(sample):
    cfg = io.load_config(ROOT / "configs" / "samples" / f"{sample}.yaml")
    base = Path(cfg["output_root"]) / sample
    meta = json.loads((base / "output" / f"{sample}_metadata.json").read_text())
    qcr = json.loads((base / "04_qc" / "qc_report.json").read_text())
    reg = json.loads((base / "03_registration" / "registration_info.json").read_text())
    return {"cfg": cfg, "base": base, "meta": meta, "qc": qcr, "reg": reg}


def out_img(S, name):
    return tifffile.imread(S["base"] / "output" / f"{S['cfg']['sample_id']}_{name}_oct_grid.tif")


def oct_u8(db, mask):
    v = np.nan_to_num(db, nan=0.0)
    lo, hi = np.percentile(v[mask], [0.5, 99.5]) if mask.any() else (0, 1)
    return np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)


def label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def show(ax, im, title=None, **kw):
    ax.imshow(im, interpolation="lanczos" if im.ndim == 3 else "nearest", **kw); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    if title: ax.set_title(title, pad=2)


def scalebar(ax, px_um, um, shape, color="w"):
    L = um / px_um; h, w = shape[:2]
    ax.plot([w * 0.95 - L, w * 0.95], [h * 0.95, h * 0.95], color=color, lw=2, solid_capstyle="butt")
    ax.text(w * 0.95 - L / 2, h * 0.93, f"{um:g} µm" if um < 1000 else f"{um / 1000:g} mm", color=color,
            ha="center", va="bottom", fontsize=6)


def ds(a, f):
    return cv2.resize(a, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def fig_pipeline(S, path):
    sid = S["cfg"]["sample_id"]; b = S["base"]
    raw_he = cv2.imread(str(b / "01_he_preproc/preview/he_00_raw.png"))[..., ::-1]
    he_n = cv2.imread(str(b / "01_he_preproc/preview/he_03_stainnorm.png"))[..., ::-1]
    raw_o = cv2.imread(str(b / "02_oct_preproc/preview/oct_00_raw.png"), 0)
    o_n = cv2.imread(str(b / "02_oct_preproc/preview/oct_03_norm.png"), 0)
    he = out_img(S, "HE_registered_stainnorm"); db = out_img(S, "OCT_dB"); m = out_img(S, "mask_common") > 0
    o = oct_u8(db, m); f = 0.2
    he_s, o_s = ds(he, f), ds(o.astype(np.float32), f)
    tp = int(500 / 2 * f)
    fig, axs = plt.subplots(1, 7, figsize=(7.2, 2.3))
    ims = [(raw_he, "H&E raw"), (he_n, "H&E bg-removed +\nMacenko"), (raw_o, "OCT raw (dB)"), (o_n, "OCT bg-removed +\nflat-field"),
           (qc.blend(he_s, o_s), "registered\nblend"), (qc.checkerboard(he_s, o_s, tp), "checkerboard A"),
           (qc.checkerboard(he_s, o_s, tp, True), "checkerboard B")]
    for i, (ax, (im, t)) in enumerate(zip(axs, ims)):
        show(ax, im, t, cmap="gray" if im.ndim == 2 else None); label(ax, "abcdefg"[i])
    scalebar(axs[4], 2 / f, 2000, he_s.shape)
    fig.tight_layout(pad=0.3, w_pad=0.4); fig.savefig(path, dpi=300); plt.close(fig)


def fig_overview(Ss, path):
    n = len(Ss)
    fig, axs = plt.subplots(n, 5, figsize=(7.2, min(1.75 * n, 9.2)))
    axs = np.atleast_2d(axs)
    for r, S in enumerate(Ss):
        he = out_img(S, "HE_registered_stainnorm"); db = out_img(S, "OCT_dB"); m = out_img(S, "mask_common") > 0
        f = 1000 / max(he.shape[:2]); he_s = ds(he, f); o = ds(oct_u8(db, m).astype(np.float32), f)
        hi = ds(qc_intensity(he), f)
        tp = max(4, int(500 / 2 * f))
        ims = [(o, "OCT (fixed)"), (he_s, "H&E registered"), (qc.checkerboard(he_s, o, tp), "checkerboard A (500 µm)"),
               (qc.checkerboard(he_s, o, tp, True), "checkerboard B (complement)"), (qc.false_color(hi, o), "OCT magenta / H&E green")]
        for c, (im, t) in enumerate(ims):
            show(axs[r, c], im, t if r == 0 else None, cmap="gray" if im.ndim == 2 else None)
        axs[r, 0].set_ylabel(S["cfg"]["sample_id"], fontsize=7); axs[r, 0].yaxis.set_visible(True); axs[r, 0].set_yticks([])
        scalebar(axs[r, 1], 2 / f, 2000, he_s.shape, "k")
    fig.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.3); fig.savefig(path, dpi=300); plt.close(fig)


def qc_intensity(he_rgb):
    od = -np.log(np.clip(he_rgb.astype(np.float32), 1, None) / 255.0).sum(-1)
    return np.clip(od / max(np.percentile(od, 99.5), 1e-6), 0, 1).astype(np.float32)


def fig_rois(Ss, path, per_sample=None, size_um=500):
    per_sample = per_sample or (2 if len(Ss) <= 4 else 1)   # keep the figure on one page
    rows = []
    for S in Ss:
        rois = S["qc"]["rois"]
        pick = ([r for r in rois if r["kind"] == "edge"][:1] + [r for r in rois if r["kind"] == "auto"][:1]) if per_sample == 1 else \
               ([r for r in rois if r["kind"] == "auto"][:1] + [r for r in rois if r["kind"] == "edge"][:1])
        for r in pick[:per_sample]:
            rows.append((S, r))
    fig, axs = plt.subplots(len(rows), 5, figsize=(7.2, 1.5 * len(rows)))
    axs = np.atleast_2d(axs)
    cache = {}
    for i, (S, r) in enumerate(rows):
        sid = S["cfg"]["sample_id"]
        if sid not in cache:
            he = out_img(S, "HE_registered_stainnorm"); db = out_img(S, "OCT_dB"); m = out_img(S, "mask_common") > 0
            cache = {sid: (he, oct_u8(db, m).astype(np.float32))}
        he, o = cache[sid]
        h = int(size_um / 2); cx, cy = int(r["cx_um"] / 2), int(r["cy_um"] / 2)
        y0, x0 = max(0, cy - h // 2), max(0, cx - h // 2)
        sl = np.s_[y0:y0 + h, x0:x0 + h]
        hs, os_ = he[sl], o[sl]; hi = qc_intensity(hs); tp = 50
        ims = [(hs, "H&E registered"), (os_, "OCT"), (qc.checkerboard(hs, os_, tp), "checkerboard A (100 µm)"),
               (qc.checkerboard(hs, os_, tp, True), "checkerboard B"), (qc.edge_overlay(hi, os_), "edges OCT / H&E")]
        for c, (im, t) in enumerate(ims):
            show(axs[i, c], im, t if i == 0 else None, cmap="gray" if im.ndim == 2 else None)
            for k in range(50, h, 50):
                axs[i, c].axhline(k, color="y", lw=0.25, alpha=0.6); axs[i, c].axvline(k, color="y", lw=0.25, alpha=0.6)
        axs[i, 0].set_ylabel(f"{sid}\n{r['kind']} ({r['cx_um']:.0f},{r['cy_um']:.0f}) µm", fontsize=6)
        axs[i, 0].yaxis.set_visible(True); axs[i, 0].set_yticks([])
        scalebar(axs[i, 1], 2, 100, os_.shape, "c")
    fig.tight_layout(pad=0.3, h_pad=0.3, w_pad=0.3); fig.savefig(path, dpi=300); plt.close(fig)


def fig_accuracy(Ss, path):
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.2))
    cmap = plt.get_cmap("tab10")
    for i, S in enumerate(Ss):
        for tag, ls in (("affine", ":"), ("deformable", "-")):
            z = np.load(S["base"] / "04_qc/residuals" / f"landmarks_{tag}.npz")
            e = np.sort(z["err_um"]); axs[0].plot(e, np.linspace(0, 1, len(e)), ls, color=cmap(i), lw=1,
                                                   label=f"{S['cfg']['sample_id']} {tag}" if tag == "deformable" else None)
    axs[0].axvline(2, color="k", lw=0.5, ls="--"); axs[0].set_xlim(0, 20)
    axs[0].set_xlabel("landmark error (µm)"); axs[0].set_ylabel("cumulative fraction"); axs[0].legend(frameon=False)
    axs[0].set_title("lumen-landmark TRE (dotted: affine)"); label(axs[0], "a")
    x = np.arange(len(Ss)); w = 0.38
    for j, tag in enumerate(("affine", "deformable")):
        med = [S["qc"][f"landmark_{tag}"]["median_um"] for S in Ss]; p90 = [S["qc"][f"landmark_{tag}"]["p90_um"] for S in Ss]
        axs[1].bar(x + (j - 0.5) * w, med, w, color=COL[tag], label=tag)
        axs[1].scatter(x + (j - 0.5) * w, p90, marker="_", color="k", s=40, zorder=3)
        rm = [S["qc"][f"residual_{tag}"]["median_um"] for S in Ss]
        axs[2].bar(x + (j - 0.5) * w, rm, w, color=COL[tag], label=tag)
    for a, t, l in ((axs[1], "landmark TRE median (bar) / p90 (tick)", "b"), (axs[2], "block-matching residual median", "c")):
        a.set_xticks(x); a.set_xticklabels([S["cfg"]["sample_id"] for S in Ss], rotation=30); a.set_ylabel("µm")
        a.axhline(2, color="k", lw=0.5, ls="--"); a.set_title(t); a.legend(frameon=False); label(a, l)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    for sp in ("top", "right"): axs[0].spines[sp].set_visible(False)
    fig.tight_layout(pad=0.4); fig.savefig(path, dpi=300); plt.close(fig)


def fig_landmarks(S, path, win_um=500, sub_um=200, zoom_um=46):
    """Explain the lumen-landmark TRE on real data: segmentation, matching, one pair, error distribution."""
    from scipy.spatial import cKDTree
    import matplotlib.patches as mpatches
    b = S["base"]; px = float(S["meta"]["grids"]["oct_grid"]["pixel_size_um"])
    o = tifffile.imread(b / "02_oct_preproc" / "oct_norm.tif")
    om = tifffile.imread(b / "03_registration" / "oct_tissue_mask_refined.tif") > 0
    hw = tifffile.imread(b / "03_registration" / "he_intensity_warped_deformable.tif")
    he = out_img(S, "HE_registered_stainnorm")
    thr = float(S["cfg"].get("qc", {}).get("oct_lumen_db", 2.6))
    odb = np.nan_to_num(tifffile.imread(b / "02_oct_preproc" / "oct_clean_db.tif"), nan=0.0).astype(np.float32)
    co, ao, mo = qc.lumen_landmarks(odb, om, thr=thr, return_mask=True)
    ch, ah, mh = qc.lumen_landmarks(hw, om, return_mask=True)
    pairs, st = qc.landmark_tre(odb, hw, om, px, oct_thr=thr)
    pa = np.load(b / "04_qc" / "residuals" / "landmarks_affine.npz")["err_um"]
    P_o, P_h, E = pairs["oct_yx_px"], pairs["he_yx_px"], pairs["err_um"]
    # window with the most matched pairs
    w = int(win_um / px); tree = cKDTree(P_o)
    cnt = [len(tree.query_ball_point(p, w / 2, p=np.inf)) for p in P_o]
    cy, cx = P_o[int(np.argmax(cnt))]
    H, W = o.shape; y0 = int(np.clip(cy - w / 2, 0, H - w)); x0 = int(np.clip(cx - w / 2, 0, W - w))
    sl = np.s_[y0:y0 + w, x0:x0 + w]
    inw = lambda P: (P[:, 0] >= y0) & (P[:, 0] < y0 + w) & (P[:, 1] >= x0) & (P[:, 1] < x0 + w)
    k = inw(P_o)

    def label(ax, t):   # panel letter above-left of the title (titles here are long)
        ax.text(-0.04, 1.13, t, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")

    def outlines(ax, m, col, lw=1.0):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = c[:, 0, :]; ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, lw=lw)

    fig = plt.figure(figsize=(7.2, 5.0))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.92], hspace=0.34, wspace=0.34)
    oc = np.clip(o[sl], 0, 1)
    # a: OCT lumens
    ax = fig.add_subplot(gs[0, 0]); show(ax, oc, "OCT: dark lumens", cmap="gray", vmin=0, vmax=1)
    outlines(ax, mo[sl], "#ff2bd6"); m = inw(co)
    ax.plot(co[m, 1] - x0, co[m, 0] - y0, "+", color="#ff2bd6", ms=4, mew=0.8); label(ax, "a")
    sw = int(sub_um / px); kk = np.flatnonzero(k)
    t2 = cKDTree(P_o[kk]); c2 = [len(t2.query_ball_point(p, sw / 2, p=np.inf)) for p in P_o[kk]]
    sy, sx = P_o[kk][int(np.argmax(c2))]
    sy0 = int(np.clip(sy - sw / 2, y0, y0 + w - sw)); sx0 = int(np.clip(sx - sw / 2, x0, x0 + w - sw))
    ax.add_patch(mpatches.Rectangle((sx0 - x0, sy0 - y0), sw, sw, fill=False, color="#ffd60a", lw=1))
    scalebar(ax, px, 100, oc.shape)
    # b: H&E lumens
    ax = fig.add_subplot(gs[0, 1]); show(ax, he[sl], "registered H&E: unstained lumens")
    outlines(ax, mh[sl], "#00b050"); m = inw(ch)
    ax.plot(ch[m, 1] - x0, ch[m, 0] - y0, "+", color="#006b2f", ms=4, mew=0.8); label(ax, "b")
    # c: matching, zoomed to the yellow box in (a) so that the few-um pair vectors are visible
    ssl = np.s_[sy0:sy0 + sw, sx0:sx0 + sw]
    insub = lambda P: (P[:, 0] >= sy0) & (P[:, 0] < sy0 + sw) & (P[:, 1] >= sx0) & (P[:, 1] < sx0 + sw)
    ks = insub(P_o)
    ax = fig.add_subplot(gs[0, 2]); show(ax, np.clip(o[ssl], 0, 1), f"matched pairs in box (n={int(ks.sum())}), vectors x5", cmap="gray", vmin=0, vmax=1)
    outlines(ax, mo[ssl], "#ff2bd6", 0.8); outlines(ax, mh[ssl], "#00b050", 0.8)
    cmap = plt.get_cmap("turbo")
    for (yo, xo), (yh, xh), e in zip(P_o[ks], P_h[ks], E[ks]):
        ex, ey = xo + 5 * (xh - xo), yo + 5 * (yh - yo)          # error vector exaggerated x5 for visibility
        ax.annotate("", xy=(ex - sx0, ey - sy0), xytext=(xo - sx0, yo - sy0),
                    arrowprops=dict(arrowstyle="-|>", color=cmap(min(e / 10, 1)), lw=1.4, mutation_scale=6))
        ax.plot(xo - sx0, yo - sy0, ".", color="#ff2bd6", ms=3)
    matched_o = {tuple(p) for p in np.round(P_o, 3)}
    un = np.array([p for p in co[insub(co)] if tuple(np.round(p, 3)) not in matched_o]) if insub(co).any() else np.zeros((0, 2))
    if len(un): ax.plot(un[:, 1] - sx0, un[:, 0] - sy0, "x", color="#ffd60a", ms=4, mew=1)
    scalebar(ax, px, 50, (sw, sw))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 10)); cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("pair error (µm)", fontsize=6); cb.ax.tick_params(labelsize=5); label(ax, "c")
    # d: one pair, how the error is computed (median-error pair in the window)
    ki = np.flatnonzero(k); j = ki[np.argsort(E[ki])[int(0.75 * (len(ki) - 1))]]   # 75th-percentile pair (visible arrow)
    (yo, xo), (yh, xh), e = P_o[j], P_h[j], E[j]
    z = int(zoom_um / px); zy0, zx0 = int(yo - z / 2), int(xo - z / 2); zs = np.s_[zy0:zy0 + z, zx0:zx0 + z]
    ax = fig.add_subplot(gs[1, 0]); show(ax, np.clip(o[zs], 0, 1), "one pair: error = centroid distance", cmap="gray", vmin=0, vmax=1)
    outlines(ax, mo[zs], "#ff2bd6", 1.4); outlines(ax, mh[zs], "#00b050", 1.4)
    ax.plot(xo - zx0, yo - zy0, "o", color="#ff2bd6", ms=5, mec="w", mew=0.6); ax.plot(xh - zx0, yh - zy0, "o", color="#00b050", ms=5, mec="w", mew=0.6)
    ax.annotate("", xy=(xh - zx0, yh - zy0), xytext=(xo - zx0, yo - zy0), arrowprops=dict(arrowstyle="-|>", color="#ffd60a", lw=1.6, mutation_scale=8))
    ax.add_patch(mpatches.Circle((xo - zx0, yo - zy0), 20 / px, fill=False, ls="--", lw=0.8, color="#00e5ff"))
    d2 = cKDTree(ch).query([yo, xo], k=2)[0][1] * px
    ax.text(0.03, 0.97, f"d = {e:.1f} µm\n2nd nearest H&E = {d2:.0f} µm\nratio {e / d2:.2f} < 0.5 ✓", transform=ax.transAxes, va="top",
            fontsize=6, color="w", bbox=dict(boxstyle="round,pad=0.25", fc=(0, 0, 0, .55), ec="none"))
    ax.text(0.03, 0.03, "cyan: 20 µm search radius", transform=ax.transAxes, fontsize=5.5, color="#00e5ff"); label(ax, "d")
    # e: rules + distribution
    ax = fig.add_subplot(gs[1, 1]); bins = np.arange(0, 20.5, 0.5)
    ax.hist(pa, bins=bins, color="#9aa5b1", alpha=.8, label=f"affine (median {np.median(pa):.2f} µm)")
    ax.hist(E, bins=bins, color="#1f6feb", alpha=.8, label=f"deformable (median {np.median(E):.2f} µm)")
    ax.axvline(2, color="k", lw=0.6, ls="--"); ax.text(2.3, ax.get_ylim()[1] * 0.55, "1 OCT\npixel", fontsize=5.5)
    ax.set_xlabel("landmark error (µm)"); ax.set_ylabel("matched lumens", labelpad=1); ax.legend(frameon=False, fontsize=5.5, loc="upper right")
    ax.set_title(f"all {len(E)} pairs, {S['cfg']['sample_id']}"); label(ax, "e")
    for sp_ in ("top", "right"): ax.spines[sp_].set_visible(False)
    # f: where the landmarks are
    ax = fig.add_subplot(gs[1, 2]); f = 0.12; bg = cv2.resize(np.clip(o, 0, 1), None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    show(ax, bg, "all landmarks (box = a)", cmap="gray", vmin=0, vmax=1)
    ax.scatter(P_o[:, 1] * f, P_o[:, 0] * f, c=np.minimum(E, 10), cmap="turbo", vmin=0, vmax=10, s=1.2, lw=0)
    ax.add_patch(mpatches.Rectangle((x0 * f, y0 * f), w * f, w * f, fill=False, color="#ffd60a", lw=1)); label(ax, "f")
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return {"n_in_view": int(k.sum()), "n_total": len(E), "pair_err": float(e), "pair_d2": float(d2)}


def fig_deformation(Ss, path, grid_um=16.0, cap_um=30.0):
    """Non-rigid displacement (on top of the affine) for every specimen: magnitude map + exaggerated grid."""
    import torch
    from heoct.register import Transform
    n = len(Ss); nc = min(n, 4); nr = int(np.ceil(n / nc))          # rows of at most 4 specimens
    fig, A = plt.subplots(2 * nr, nc, figsize=(7.2, 2.3 * 2 * nr), gridspec_kw={"hspace": 0.18, "wspace": 0.08}, squeeze=False)
    stats = []; mag_axes = []
    for ax in A.ravel(): ax.set_axis_off()
    for i, S in enumerate(Ss):
        r, c = divmod(i, nc); axs = {0: {i: A[2 * r, c]}, 1: {i: A[2 * r + 1, c]}}; A[2 * r, c].set_axis_on(); A[2 * r + 1, c].set_axis_on()
        b = S["base"]; T = Transform.load(b / "03_registration" / "transform_full.npz")
        X = T.grid_um(grid_um)
        with torch.no_grad():
            u = T.displacement(X).cpu().numpy()
        Xn = X.cpu().numpy(); mag = np.hypot(u[..., 0], u[..., 1])
        om = tifffile.imread(b / "03_registration" / "oct_tissue_mask_refined.tif") > 0
        m = cv2.resize(om.astype(np.uint8), (mag.shape[1], mag.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        ext = (0, T.ext_um[1] / 1000, T.ext_um[0] / 1000, 0)
        ax = axs[0][i]; mag_axes.append(ax); im = ax.imshow(np.where(m, mag, np.nan), cmap="magma", vmin=0, vmax=cap_um, extent=ext)
        ax.set_facecolor("black"); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(S["cfg"]["sample_id"], pad=2)
        st = 24; q = np.s_[st // 2::st, st // 2::st]; mq = m[q]
        ax.quiver(Xn[..., 0][q][mq] / 1000, Xn[..., 1][q][mq] / 1000, u[..., 0][q][mq], -u[..., 1][q][mq], color="w",
                  angles="xy", scale_units="xy", scale=1000 / 10, width=0.009, headwidth=4)     # arrows x10
        ax = axs[1][i]; ax.set_facecolor("white")
        gx, gy = (Xn[..., 0] + 10 * u[..., 0]) / 1000, (Xn[..., 1] + 10 * u[..., 1]) / 1000
        gs_ = 12
        for k in range(0, gx.shape[0], gs_):
            ax.plot(np.where(m[k], gx[k], np.nan), np.where(m[k], gy[k], np.nan), color="#3b1f4a", lw=0.45)
        for k in range(0, gx.shape[1], gs_):
            ax.plot(np.where(m[:, k], gx[:, k], np.nan), np.where(m[:, k], gy[:, k], np.nan), color="#3b1f4a", lw=0.45)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        mm = mag[m]; stats.append({"sample": S["cfg"]["sample_id"], "median_um": float(np.median(mm)), "p99_um": float(np.percentile(mm, 99)),
                                   "max_um": float(mm.max()), "frac_gt50": float((mm > 50).mean())})
    for r in range(nr):
        A[2 * r, 0].set_ylabel("|u| (µm)"); A[2 * r + 1, 0].set_ylabel("grid ×10")
    for ax, t in ((A[0, 0], "a"), (A[1, 0], "b")):
        ax.text(-0.08, 1.02, t, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")
    cb = fig.colorbar(im, ax=mag_axes, fraction=0.02, pad=0.01, extend="max"); cb.set_label("displacement (µm)", fontsize=6); cb.ax.tick_params(labelsize=5)
    fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)
    return stats


def f2(v, d=2):
    return "--" if v is None else f"{v:.{d}f}"


def tables(Ss):
    t1 = []
    for S in Ss:
        q, m = S["qc"], S["meta"]
        la, ld = q["landmark_affine"], q["landmark_deformable"]
        ra, rd = q["residual_affine"], q["residual_deformable"]
        t1.append(f"{S['cfg']['sample_id'].replace('_', chr(92) + '_')} & {f2(q['mask_dice_affine'], 3)} & {f2(q['mask_dice_deformable'], 3)} & "
                  f"{f2((q.get('boundary_deformable') or {}).get('median_um'), 1)} & "
                  f"{ld['n_matched']} & {f2(la['median_um'])} & {f2(ld['median_um'])} & {f2(ld['p90_um'])} & {100 * ld['frac_below_2um']:.0f} & "
                  f"{100 * ld['frac_below_4um']:.0f} & {f2(ra['median_um'])} & {f2(rd['median_um'])} & {f2(rd['p90_um'])} \\\\")
    t2 = []
    for S in Ss:
        m, q = S["meta"], S["qc"]; ori = q["orientation"]; hist = S["reg"]["history"]
        dfin = [h for h in hist if h["stage"] == "deform"][-1]
        t2.append(f"{S['cfg']['sample_id'].replace('_', chr(92) + '_')} & {m['inputs']['he']['shape'][1]}$\\times${m['inputs']['he']['shape'][0]} & "
                  f"{m['inputs']['oct']['shape'][1]}$\\times${m['inputs']['oct']['shape'][0]} & {m['inputs']['he']['pixel_size_um_estimated']:.3f} & "
                  f"{'yes' if ori['mirrored'] else 'no'} & {ori['rotation_deg']:.1f} & {q['deformation']['median_um']:.1f} / {q['deformation']['max_um']:.1f} & "
                  f"{dfin['jac_min']:.2f}--{dfin['jac_max']:.2f} & {100 * dfin['fold_frac']:.2f} & {S['reg']['runtime_s'] / 60:.1f} \\\\")
    return "\n".join(t1), "\n".join(t2)


def pooled(Ss, tag):
    e = np.concatenate([np.load(S["base"] / "04_qc/residuals" / f"landmarks_{tag}.npz")["err_um"] for S in Ss])
    return {"n": len(e), "median": float(np.median(e)), "p90": float(np.percentile(e, 90)),
            "lt2": float((e < 2).mean()), "lt4": float((e < 4).mean()), "mean": float(e.mean())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--samples", nargs="*"); a = ap.parse_args()
    samples = a.samples or [p.stem for p in sorted((ROOT / "configs" / "samples").glob("*.yaml"))]
    Ss = [load(s) for s in samples]
    rep = io.aux_root(Ss[0]["cfg"]) / "report"; fd = rep / "figures"; fd.mkdir(parents=True, exist_ok=True)
    print("figures ..."); fig_pipeline(Ss[0], fd / "fig1_pipeline.png"); fig_overview(Ss, fd / "fig2_overview.png")
    fig_rois(Ss, fd / "fig3_rois.png"); fig_accuracy(Ss, fd / "fig4_accuracy.png")
    lm = fig_landmarks(Ss[0], fd / "fig5_landmarks.png")
    dstats = fig_deformation(Ss, fd / "fig6_deformation.png")
    (rep / "deformation_stats.json").write_text(json.dumps(dstats, indent=1))
    t1, t2 = tables(Ss); pa, pd = pooled(Ss, "affine"), pooled(Ss, "deformable")
    rd_med = np.median([S["qc"]["residual_deformable"]["median_um"] for S in Ss])
    dice = [S["qc"]["mask_dice_deformable"] for S in Ss]
    tpl = (ROOT / "report_template" / "report.tex").read_text()
    subs = {"NSAMPLES": str(len(Ss)), "SAMPLELIST": ", ".join(s.replace("_", r"\_") for s in samples),
            "TABLE1": t1, "TABLE2": t2, "DATE": time.strftime("%d %B %Y"),
            "POOLN": str(pd["n"]), "POOLAFFMED": f"{pa['median']:.2f}", "POOLAFFP90": f"{pa['p90']:.1f}",
            "POOLDEFMED": f"{pd['median']:.2f}", "POOLDEFP90": f"{pd['p90']:.1f}", "POOLDEFLT2": f"{100 * pd['lt2']:.0f}",
            "POOLDEFLT4": f"{100 * pd['lt4']:.0f}", "POOLAFFLT2": f"{100 * pa['lt2']:.0f}", "RESDEFMED": f"{rd_med:.2f}",
            "DICEMIN": f"{min(dice):.3f}", "DICEMAX": f"{max(dice):.3f}",
            "DEFMEDS": ", ".join(f"{d['median_um']:.1f}" for d in dstats), "DEFP99MAX": f"{max(d['p99_um'] for d in dstats):.0f}",
            "DEFGT50": f"{100 * max(d['frac_gt50'] for d in dstats):.1f}",
            "LMSAMPLE": Ss[0]["cfg"]["sample_id"].replace("_", r"\_"), "LMNTOT": str(lm["n_total"]), "LMNVIEW": str(lm["n_in_view"]),
            "EDGEMED": f"{np.median([(S['qc'].get('boundary_deformable') or {}).get('median_um', np.nan) for S in Ss]):.1f}",
            "HEPX": ", ".join(f"{S['meta']['inputs']['he']['pixel_size_um_estimated']:.3f}" for S in Ss),
            "MIRRORED": ", ".join(S["cfg"]["sample_id"].replace("_", r"\_") for S in Ss if S["qc"]["orientation"]["mirrored"]) or "none"}
    for k, v in subs.items():
        tpl = tpl.replace(f"<<{k}>>", v)
    (rep / "report.tex").write_text(tpl)
    (rep / "report_numbers.json").write_text(json.dumps({"samples": samples, "pooled_affine": pa, "pooled_deformable": pd, **{k: v for k, v in subs.items() if not k.startswith("TABLE")}}, indent=1))
    print("compiling ...")
    tect = shutil.which("tectonic") or next((str(c) for c in (Path(sys.executable).parent / "tectonic",
                                                             Path(sys.executable).parent / "Library" / "bin" / "tectonic.exe")
                                            if c.exists()), None)
    if not tect:
        print("tectonic not found: report.tex written but not compiled (conda install -c conda-forge tectonic)"); return
    r = subprocess.run([tect, "report.tex"], cwd=rep, capture_output=True, text=True)
    print(r.stdout[-1500:], r.stderr[-3000:])
    print("PDF:", rep / "report.pdf", (rep / "report.pdf").exists())


if __name__ == "__main__":
    main()
