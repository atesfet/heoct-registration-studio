#!/usr/bin/env python
"""Cross-sample check that H&E stain normalisation and OCT standardisation are consistent.

a  OCT intensity distribution inside the tissue after standardisation (should overlap, median at 0.5)
b  OCT median and IQR in dB per sample (raw scale differences that standardisation removes)
c  fitted hematoxylin / eosin 99th-percentile concentrations (outliers -> washed-out normalisation)
d  angle between each fitted stain vector and the reference (a large H angle = fit pulled by ink/debris)
e  stain-normalised H&E thumbnails, same reference for every sample

Usage: python scripts/normalization_check.py [--root <results root>] [sample ...]
Output: <aux_root>/figures/normalization_check/normalization_check.png (+ .json with the numbers)
"""
import argparse, json, sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import yaml

REF_H = np.array([0.5626, 0.7201, 0.4062]); REF_E = np.array([0.2159, 0.8012, 0.5581])


def angle(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("samples", nargs="*"); a = ap.parse_args()
    d = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text())
    root = Path(a.root or d["output_root"]); aux = Path(d.get("aux_root") or root)
    samples = a.samples or sorted(x.name for x in root.iterdir() if x.is_dir() and any((x / "output").glob("*_metadata.json")))
    rows, hists, thumbs = [], {}, []
    for s in samples:
        b = root / s
        oi = json.loads((b / "02_oct_preproc" / "oct_preproc_info.json").read_text())
        hi = json.loads((b / "01_he_preproc" / "he_preproc_info.json").read_text())
        n = tifffile.imread(b / "02_oct_preproc" / "oct_norm.tif"); m = tifffile.imread(b / "02_oct_preproc" / "oct_tissue_mask.tif") > 0
        hists[s] = np.histogram(n[m], bins=np.linspace(0, 1, 51))[0] / max(m.sum(), 1)
        st = oi.get("standardization", {}); M = np.array(hi["stain_matrix_he_columns"])
        rows.append({"sample": s, "oct_median_db": st.get("median_db"), "oct_iqr_db": st.get("iqr_db"), "oct_norm_median": st.get("norm_median"),
                     "oct_clipped_high": st.get("frac_clipped_high"), "oct_clipped_low": st.get("frac_clipped_low"),
                     "maxC_H": hi["max_concentration"][0], "maxC_E": hi["max_concentration"][1],
                     "H_angle_deg": angle(M[:, 0], REF_H), "E_angle_deg": angle(M[:, 1], REF_E),
                     "stain_fit_outlier_frac": hi.get("stain_fit_outlier_frac")})
        t = cv2.imread(str(b / "01_he_preproc" / "preview" / "he_03_stainnorm.png"))[..., ::-1]
        thumbs.append(cv2.resize(t, (int(t.shape[1] * 220 / t.shape[0]), 220), interpolation=cv2.INTER_AREA))
    out = aux / "figures" / "normalization_check"; out.mkdir(parents=True, exist_ok=True)
    (out / "normalization_check.json").write_text(json.dumps(rows, indent=1))

    fig = plt.figure(figsize=(12, 7.5)); gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.8], hspace=0.35, wspace=0.35)
    cmap = plt.get_cmap("tab10"); x = np.arange(len(rows)); names = [r["sample"] for r in rows]
    ax = fig.add_subplot(gs[0, 0]); c = np.linspace(0.01, 0.99, 50)
    for i, s in enumerate(samples): ax.plot(c, hists[s], color=cmap(i % 10), lw=1.2, label=s)
    ax.axvline(0.5, color="k", ls="--", lw=0.7); ax.set_xlabel("standardised OCT intensity"); ax.set_ylabel("fraction of tissue")
    ax.set_title("a  OCT after standardisation", loc="left"); ax.legend(fontsize=7, frameon=False)
    ax = fig.add_subplot(gs[0, 1]); ax.bar(x, [r["oct_median_db"] for r in rows], color="#56636f")
    ax.errorbar(x, [r["oct_median_db"] for r in rows], yerr=[np.array([r["oct_iqr_db"] for r in rows]) / 2] * 2, fmt="none", ecolor="k", capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax.set_ylabel("dB above noise floor"); ax.set_title("b  OCT median ± IQR/2 (raw)", loc="left")
    ax = fig.add_subplot(gs[0, 2]); w = 0.38
    ax.bar(x - w / 2, [r["maxC_H"] for r in rows], w, color="#5b3f8c", label="H"); ax.bar(x + w / 2, [r["maxC_E"] for r in rows], w, color="#c2447a", label="E")
    ax.axhline(1.9705, color="#5b3f8c", ls=":", lw=0.8); ax.axhline(1.0308, color="#c2447a", ls=":", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax.set_ylabel("99th-pct concentration"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("c  stain concentration (dotted = reference)", loc="left")
    ax = fig.add_subplot(gs[0, 3]); ax.bar(x - w / 2, [r["H_angle_deg"] for r in rows], w, color="#5b3f8c", label="H"); ax.bar(x + w / 2, [r["E_angle_deg"] for r in rows], w, color="#c2447a", label="E")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8); ax.set_ylabel("degrees from reference vector"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("d  stain-vector angle", loc="left")
    ax = fig.add_subplot(gs[1, :]); strip = np.hstack([np.pad(t, ((0, 0), (0, 10), (0, 0)), constant_values=255) for t in thumbs])
    ax.imshow(strip); ax.set_axis_off(); ax.set_title("e  stain-normalised H&E (same reference)", loc="left")
    xs = np.cumsum([0] + [t.shape[1] + 10 for t in thumbs[:-1]])
    for xi, t, s in zip(xs, thumbs, samples): ax.text(xi + t.shape[1] / 2, strip.shape[0] + 15, s, ha="center", va="top", fontsize=8)
    fig.savefig(out / "normalization_check.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    print(out / "normalization_check.png")
    for r in rows: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
