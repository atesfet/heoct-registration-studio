#!/usr/bin/env python
"""Validate registered samples as a machine-learning dataset and write one manifest.

For every registered sample (<root>/<sample>/output/) on the OCT grid it checks:
  files      all expected TIFFs + metadata present
  geometry   identical shape across modalities, pixel size == the common pixel size, same frame
  dtypes     OCT float32, H&E uint8 RGB, masks uint8 {0,255}
  values     OCT dB finite inside the OCT mask and NaN outside; standardised OCT in [0,1];
             H&E white (255) outside the registered H&E mask; common mask = OCT mask AND H&E mask
  QC         landmark TRE, tissue-edge distance, Dice, landmark count, mask refinement, folding
and flags samples that should be reviewed before training (thresholds below, editable with --thresholds).

Usage: python scripts/validate_dataset.py [--root <results root>] [sample ...]
Output: <aux_root>/dataset/{dataset_manifest.csv, dataset_manifest.json, dataset_validation.txt}
"""
import argparse, csv, json, sys
from pathlib import Path

import numpy as np
import tifffile
import yaml

THRESH = {"tre_median_um_max": 4.0, "tre_p90_um_max": 12.0, "edge_median_um_max": 10.0, "dice_min": 0.98,
          "landmarks_min": 50, "mask_refinement_removed_max": 0.05, "fold_frac_max": 0.0}
CONDITION = {"sample_13": "acetic-acid test: control", "sample_14": "acetic-acid test: acetic acid"}   # experiment notes


def read(p):
    with tifffile.TiffFile(p) as t:
        return t.asarray(), (t.imagej_metadata or {})


def check_sample(d: Path, sid: str):
    o = d / "output"; issues, warn = [], []
    need = {k: o / f"{sid}_{k}_oct_grid.tif" for k in ("OCT_dB", "OCT_standardized", "HE_registered_stainnorm", "HE_registered_original",
                                                      "mask_common", "mask_OCT", "mask_HE_registered")}
    mj = o / f"{sid}_metadata.json"
    missing = [k for k, p in need.items() if not p.exists()] + ([] if mj.exists() else ["metadata.json"])
    if missing:
        return {"sample": sid, "status": "FAIL", "issues": [f"missing: {', '.join(missing)}"]}, None
    meta = json.loads(mj.read_text())
    A = {}; px = {}
    for k, p in need.items():
        A[k], ij = read(p); px[k] = float(ij.get("pixel_size_um", "nan"))
    shapes = {k: a.shape[:2] for k, a in A.items()}
    if len(set(shapes.values())) != 1: issues.append(f"shape mismatch {shapes}")
    if len(set(px.values())) != 1: issues.append(f"pixel size mismatch {px}")
    pxu = px["OCT_dB"]
    exp_dtype = {"OCT_dB": np.float32, "OCT_standardized": np.float32, "HE_registered_stainnorm": np.uint8, "HE_registered_original": np.uint8,
                 "mask_common": np.uint8, "mask_OCT": np.uint8, "mask_HE_registered": np.uint8}
    for k, t in exp_dtype.items():
        if A[k].dtype != t: issues.append(f"{k} dtype {A[k].dtype} (expected {np.dtype(t)})")
    for k in ("HE_registered_stainnorm", "HE_registered_original"):
        if A[k].ndim != 3 or A[k].shape[2] != 3: issues.append(f"{k} not RGB: {A[k].shape}")
    m_c, m_o, m_h = (A[k] > 0 for k in ("mask_common", "mask_OCT", "mask_HE_registered"))
    for k in ("mask_common", "mask_OCT", "mask_HE_registered"):
        u = np.unique(A[k]);
        if not set(u.tolist()) <= {0, 255}: issues.append(f"{k} values {u[:6]} not in {{0,255}}")
    if not np.array_equal(m_c, m_o & m_h): issues.append("common mask != OCT mask AND H&E mask")
    db = A["OCT_dB"]
    if not np.isfinite(db[m_o]).all(): issues.append(f"OCT_dB has {(~np.isfinite(db[m_o])).sum()} non-finite px inside the OCT mask")
    if np.isfinite(db[~m_o]).any(): warn.append(f"OCT_dB finite outside the OCT mask ({np.isfinite(db[~m_o]).mean():.2%})")
    st = A["OCT_standardized"]
    if np.isnan(st).any() or st.min() < 0 or st.max() > 1: issues.append(f"OCT_standardized outside [0,1] or NaN (min {np.nanmin(st):.3f}, max {np.nanmax(st):.3f})")
    if (st[~m_o] != 0).any(): warn.append("OCT_standardized non-zero outside the OCT mask")
    he = A["HE_registered_stainnorm"]
    if (he[~m_h] != 255).mean() > 0.001: warn.append(f"H&E not white outside its mask ({(he[~m_h] != 255).any(-1).mean():.2%} px)")
    # QC from metadata
    acc = meta["accuracy"]; lt = acc["landmark_tre"]["deformable"]; ed = (acc.get("tissue_edge_distance") or {}).get("deformable") or {}
    stdz = (meta["preprocessing"]["oct"].get("standardization") or {})
    reg = json.loads((d / "03_registration" / "registration_info.json").read_text())
    dfin = [h for h in reg["history"] if h["stage"] == "deform"][-1]
    removed = reg["affine"].get("mask_refinement_removed_frac", 0.0)
    row = {"sample": sid, "condition": CONDITION.get(sid, ""), "pixel_size_um": pxu, "rows": shapes["OCT_dB"][0], "cols": shapes["OCT_dB"][1],
           "common_mask_mm2": float(m_c.sum() * pxu * pxu / 1e6), "common_mask_frac_of_oct": float(m_c.sum() / max(m_o.sum(), 1)),
           "tre_median_um": lt.get("median_um"), "tre_p90_um": lt.get("p90_um"), "landmarks": lt.get("n_matched"),
           "edge_median_um": ed.get("median_um"), "dice": acc["mask_dice"]["deformable"], "mask_refinement_removed": removed,
           "fold_frac": dfin["fold_frac"], "he_px_um_est": meta["inputs"]["he"]["pixel_size_um_estimated"],
           "mirrored": reg["init"]["best_flip"] < 0, "oct_median_db": stdz.get("median_db"), "oct_iqr_db": stdz.get("iqr_db"),
           "stain_fit_outlier_frac": meta["preprocessing"]["he"].get("stain_fit_outlier_frac"),
           "dir": str(o)}
    T = THRESH
    flags = []
    if row["tre_median_um"] is not None and row["tre_median_um"] > T["tre_median_um_max"]: flags.append(f"TRE median {row['tre_median_um']:.2f} > {T['tre_median_um_max']}")
    if row["tre_p90_um"] is not None and row["tre_p90_um"] > T["tre_p90_um_max"]: flags.append(f"TRE p90 {row['tre_p90_um']:.1f} > {T['tre_p90_um_max']}")
    if row["landmarks"] is not None and row["landmarks"] < T["landmarks_min"]: flags.append(f"only {row['landmarks']} landmarks (accuracy weakly validated)")
    if row["edge_median_um"] is not None and row["edge_median_um"] > T["edge_median_um_max"]: flags.append(f"edge distance {row['edge_median_um']:.1f} > {T['edge_median_um_max']}")
    if row["dice"] < T["dice_min"]: flags.append(f"Dice {row['dice']:.3f} < {T['dice_min']}")
    if removed > T["mask_refinement_removed_max"]: flags.append(f"{removed:.1%} of OCT tissue has no H&E")
    if row["fold_frac"] > T["fold_frac_max"]: flags.append("deformation folds")
    if row["condition"]: warn.append(f"experimental condition: {row['condition']}")
    row["status"] = "FAIL" if issues else ("REVIEW" if flags else "OK")
    row["issues"] = issues; row["flags"] = flags; row["warnings"] = warn
    return row, A


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root"); ap.add_argument("samples", nargs="*"); a = ap.parse_args()
    d = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text())
    root = Path(a.root or d["output_root"]); aux = Path(d.get("aux_root") or root)
    samples = a.samples or sorted(x.name for x in root.iterdir() if x.is_dir() and any((x / "output").glob("*_metadata.json")))
    rows = []
    for s in samples:
        r, _ = check_sample(root / s, s); rows.append(r)
        print(f"{s:10s} {r['status']:6s}", "; ".join(r.get("issues", []) + r.get("flags", [])))
    # cross-sample consistency
    cross = []
    pxs = {r["pixel_size_um"] for r in rows if "pixel_size_um" in r}
    if len(pxs) > 1: cross.append(f"pixel size differs across samples: {sorted(pxs)}")
    he_px = [r["he_px_um_est"] for r in rows if "he_px_um_est" in r]
    if he_px and (max(he_px) - min(he_px)) / min(he_px) > 0.05:
        cross.append(f"H&E native pixel size spans {min(he_px):.3f}-{max(he_px):.3f} um/px (>5%); use the *_oct_grid files (common {sorted(pxs)} um/px)")
    out = aux / "dataset"; out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in rows[0] if k not in ("issues", "flags", "warnings")] + ["issues", "flags", "warnings"]
    with open(out / "dataset_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k: ("; ".join(r[k]) if isinstance(r.get(k), list) else r.get(k)) for k in keys})
    (out / "dataset_manifest.json").write_text(json.dumps({"thresholds": THRESH, "cross_sample": cross, "samples": rows}, indent=1, default=str))
    lines = [f"{r['sample']}: {r['status']}" + "".join(f"\n    - {x}" for x in r.get("issues", []) + r.get("flags", []) + r.get("warnings", [])) for r in rows]
    (out / "dataset_validation.txt").write_text("\n".join(lines + ["", "cross-sample:"] + [f"  - {c}" for c in cross] or ["  none"]) + "\n")
    print("cross-sample:", cross or "consistent"); print("written:", out)


if __name__ == "__main__":
    main()
