#!/usr/bin/env python
"""Build a Nature-Methods-style LaTeX report (and PDF) over all registered samples.

Usage: python scripts/make_report.py [--samples 12umA sample_10 ...]
Output: <output_root>/report/{report.tex, report.pdf, figures/*.png}
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
    fig, axs = plt.subplots(n, 5, figsize=(7.2, 1.75 * n))
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


def fig_rois(Ss, path, per_sample=2, size_um=500):
    rows = []
    for S in Ss:
        rois = S["qc"]["rois"]
        pick = [r for r in rois if r["kind"] == "auto"][:1] + [r for r in rois if r["kind"] == "edge"][:1]
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
    rep = Path(Ss[0]["cfg"]["output_root"]) / "report"; fd = rep / "figures"; fd.mkdir(parents=True, exist_ok=True)
    print("figures ..."); fig_pipeline(Ss[0], fd / "fig1_pipeline.png"); fig_overview(Ss, fd / "fig2_overview.png")
    fig_rois(Ss, fd / "fig3_rois.png"); fig_accuracy(Ss, fd / "fig4_accuracy.png")
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
