#!/usr/bin/env python
"""H&E -> OCT registration pipeline runner.

Usage:
    python scripts/run_pipeline.py --sample 12umA                 # all stages, one sample
    python scripts/run_pipeline.py --all                          # every configs/samples/*.yaml
    python scripts/run_pipeline.py --sample 12umA --stages 04 05  # selected stages

Stages (outputs under <output_root>/<sample_id>/, output_root set in configs/default.yaml):
    01_he_preproc   tissue mask, background removal, Macenko stain normalisation
    02_oct_preproc  robust OCT background/debris removal, tile flat-field, normalisation
    03_registration init search -> affine -> deformable (OCT is fixed, H&E is warped onto it)
    04_qc           overlays, checkerboards, zoomed ROI panels, landmark TRE, residual map, HTML report
    05 -> output/   final registered OCT / H&E / common mask TIFFs + single metadata JSON
    06_figures      optional (figures.save_fullres_previews, default off): full-resolution Fig. 1 panels (a-g) + Fig. 2 views (h-j) + composite
"""
import argparse
import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Apple GPU: unsupported ops fall back to CPU
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from heoct import io  # noqa: E402


def stage01(cfg):
    from heoct import he_preproc
    out = io.stage_dir(cfg, "01_he_preproc"); log = io.get_logger(out, "01_he_preproc")
    log.info(f"loading {cfg['he_path']}")
    rgb = io.load_he(cfg["he_path"])
    log.info(f"H&E shape {rgb.shape} dtype {rgb.dtype}")
    r = he_preproc.run(rgb, cfg["he"])
    log.info(f"info {r['info']}")
    tifffile.imwrite(out / "he_tissue_mask.tif", (r["mask"] * 255).astype(np.uint8), compression="zlib")
    tifffile.imwrite(out / "he_bg_removed_rgb.tif", r["bg_removed"], compression="zlib")
    tifffile.imwrite(out / "he_stainnorm_rgb.tif", r["stain_norm"], compression="zlib")
    tifffile.imwrite(out / "he_hematoxylin_conc.tif", r["hema"].astype(np.float16), compression="zlib")
    tifffile.imwrite(out / "he_eosin_conc.tif", r["eosin"].astype(np.float16), compression="zlib")
    pv = out / "preview"; pv.mkdir(exist_ok=True)
    io.save_png(pv / "he_00_raw.png", rgb, 2000)
    io.save_png(pv / "he_01_tissue_mask.png", (r["mask"] * 255).astype(np.uint8), 2000)
    io.save_png(pv / "he_02_bg_removed.png", r["bg_removed"], 2000)
    io.save_png(pv / "he_03_stainnorm.png", r["stain_norm"], 2000)
    io.save_png(pv / "he_04_hematoxylin.png", io.to_uint8(r["hema"], 0, np.percentile(r["hema"][r["mask"]], 99.5)), 2000)
    io.save_png(pv / "he_05_eosin.png", io.to_uint8(r["eosin"], 0, np.percentile(r["eosin"][r["mask"]], 99.5)), 2000)
    io.write_json(out / "he_preproc_info.json", {"input": cfg["he_path"], "shape": rgb.shape, "params": cfg["he"], **r["info"]})
    log.info("done")


def stage02(cfg):
    from heoct import oct_preproc
    out = io.stage_dir(cfg, "02_oct_preproc"); log = io.get_logger(out, "02_oct_preproc")
    img, meta = io.load_oct(cfg["oct_path"])
    log.info(f"OCT shape {img.shape}, px {meta['px_um']} um, NaN frac {np.isnan(img).mean():.3f}")
    r = oct_preproc.run(img, cfg["oct"] | {"px_um": meta["px_um"]})
    log.info(f"info {r['info']}")
    px = meta["px_um"]
    tifffile.imwrite(out / "oct_tissue_mask.tif", (r["mask"] * 255).astype(np.uint8), compression="zlib")
    io.save_tif(out / "oct_clean_db.tif", r["clean_db"], px, {"value": "dB above noise floor, flat-fielded, NaN=background"})
    io.save_tif(out / "oct_norm.tif", r["norm"], px, {"value": "0-1 normalised, 0=background"})
    tifffile.imwrite(out / "oct_flatfield_pattern_db.tif", r["flatfield"])
    pv = out / "preview"; pv.mkdir(exist_ok=True)
    lo, hi = np.nanpercentile(img, [1, 99.5])
    io.save_png(pv / "oct_00_raw.png", io.to_uint8(np.nan_to_num(img, nan=0), lo, hi), 2000)
    io.save_png(pv / "oct_01_tissue_mask.png", (r["mask"] * 255).astype(np.uint8), 2000)
    io.save_png(pv / "oct_02_removed_debris.png", ((np.isfinite(img) & ~r["mask"]) * 255).astype(np.uint8), 2000)
    io.save_png(pv / "oct_03_norm.png", (r["norm"] * 255).astype(np.uint8), 2000)
    ff = r["flatfield"]
    io.save_png(pv / "oct_04_flatfield_pattern.png", io.to_uint8(ff, ff.min(), ff.max()))
    io.write_json(out / "oct_preproc_info.json", {"input": cfg["oct_path"], "shape": img.shape, "px_um": px,
                                                 "params": cfg["oct"], "oct_metadata": meta, **r["info"]})
    log.info("done")


def he_intensity_for_registration(stain_rgb, mask, sn_cfg=None):
    """Total optical density of stain-normalised H&E, scaled to ~[0,1]; 0 = background.

    The scale (99.5th percentile) is taken over stain pixels only: ink, debris and folds (he_preproc.stain_outliers)
    are excluded from it and simply saturate at 1, so they cannot compress the rest of the tissue.
    """
    from heoct.he_preproc import stain_outliers
    od = -np.log(np.clip(stain_rgb.astype(np.float32), 1, None) / 255.0).sum(-1)
    od[~mask] = 0
    ref = mask.copy()
    ref[mask] = ~stain_outliers(stain_rgb[mask], np.array([255.0, 255.0, 255.0]), sn_cfg or {})
    return np.clip(od / np.percentile(od[ref if ref.any() else mask], 99.5), 0, 1).astype(np.float32)


def stage03(cfg):
    import torch
    from heoct import register as R
    out = io.stage_dir(cfg, "03_registration"); log = io.get_logger(out, "03_registration")
    from heoct.device import device_name
    log.info(f"compute device: {device_name()}  (NVIDIA GPU: minutes; Apple GPU / CPU: much slower, see README)")
    d1, d2 = io.stage_dir(cfg, "01_he_preproc"), io.stage_dir(cfg, "02_oct_preproc")
    he_rgb = tifffile.imread(d1 / "he_stainnorm_rgb.tif")
    he_mask = tifffile.imread(d1 / "he_tissue_mask.tif") > 0
    oct_norm = tifffile.imread(d2 / "oct_norm.tif")
    oct_mask = tifffile.imread(d2 / "oct_tissue_mask.tif") > 0
    px = float(json.loads((d2 / "oct_preproc_info.json").read_text())["px_um"])
    he_int = he_intensity_for_registration(he_rgb, he_mask, cfg["he"]["stain_norm"])
    tifffile.imwrite(out / "he_registration_intensity.tif", he_int, compression="zlib")
    rc = cfg["registration"]
    reg = R.Registrar(oct_norm, oct_mask, px, he_int, he_mask, rc, log)
    t0 = time.time()
    init = reg.init_search()
    met_init = reg.metrics(8.0, with_deform=False)
    aff = reg.affine()
    # cross-modal mask refinement: drop OCT "tissue" that has no H&E counterpart (neighbouring pieces, debris,
    # mounting medium), then refine the affine with the consensus mask
    import cv2
    hm_aff = R.warp_image(he_mask.astype(np.float32), reg.T, False) > 0.5
    rr = int(round(rc.get("mask_refine_dilate_um", 100) / px))
    oct_mask_ref = oct_mask & (cv2.dilate(hm_aff.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1))) > 0)
    removed = float((oct_mask & ~oct_mask_ref).sum() / oct_mask.sum())
    log.info(f"mask refinement removed {100 * removed:.2f}% of OCT tissue mask (no H&E counterpart)")
    tifffile.imwrite(out / "oct_tissue_mask_refined.tif", (oct_mask_ref * 255).astype(np.uint8), compression="zlib")
    reg.set_fixed_mask(oct_mask_ref)
    if removed > 0.005:
        aff = reg.affine()
    aff["mask_refinement_removed_frac"] = removed
    met_aff = reg.metrics(8.0, with_deform=False)
    reg.T.save(out / "transform_affine.npz")
    reg.deform()
    met_def = reg.metrics(8.0, with_deform=True)
    reg.T.save(out / "transform_full.npz")
    log.info(f"metrics @8um  init {met_init}  affine {met_aff}  deform {met_def}  ({time.time() - t0:.0f}s)")

    T = reg.T
    np.save(out / "map_oct_px_to_he_px.npy", R.dense_map_px(T, True))
    for tag, wd in (("affine", False), ("deformable", True)):
        tifffile.imwrite(out / f"he_stainnorm_warped_{tag}.tif", R.warp_image(he_rgb, T, wd, fill=255.0), compression="zlib")
        tifffile.imwrite(out / f"he_mask_warped_{tag}.tif",
                         ((R.warp_image(he_mask.astype(np.float32), T, wd) > 0.5) * 255).astype(np.uint8), compression="zlib")
    bg = tifffile.imread(d1 / "he_bg_removed_rgb.tif")
    tifffile.imwrite(out / "he_bg_removed_warped_deformable.tif", R.warp_image(bg, T, True, fill=255.0), compression="zlib")
    tifffile.imwrite(out / "he_intensity_warped_deformable.tif", R.warp_image(he_int, T, True), compression="zlib")
    io.write_json(out / "registration_info.json", {
        "fixed": "OCT (" + cfg["oct_path"] + ")", "moving": "H&E (" + cfg["he_path"] + ")",
        "output_grid": f"OCT pixel grid {oct_norm.shape} at {px} um/px",
        "transform_definition": R.__doc__, "params": rc, "init": init, "affine": aff,
        "metrics_8um": {"init": met_init, "affine": met_aff, "deformable": met_def},
        "history": reg.history, "runtime_s": time.time() - t0,
        "files": {"transform_full.npz": "Transform.load() -> map_px / he_px_to_oct_um",
                  "map_oct_px_to_he_px.npy": "(2,H,W) float32: H&E (x,y) pixel for every OCT pixel"}})
    log.info("done")


def stage04(cfg):
    from heoct import qc
    qc.run(cfg)


def stage05(cfg):
    from heoct import export
    export.run(cfg)


def stage06(cfg):
    from heoct import figures
    figures.run(cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="path to a per-sample yaml")
    g.add_argument("--sample", help="sample id (configs/samples/<id>.yaml)")
    g.add_argument("--all", action="store_true", help="run every configs/samples/*.yaml")
    ap.add_argument("--stages", nargs="+", default=None,
                    help="default: 01-05, plus 06 when figures.save_fullres_previews is true")
    ap.add_argument("--default", help="default config to merge under the sample config (default: configs/default.yaml)")
    a = ap.parse_args()
    cdir = Path(__file__).resolve().parents[1] / "configs" / "samples"
    paths = sorted(cdir.glob("*.yaml")) if a.all else [Path(a.config) if a.config else cdir / f"{a.sample}.yaml"]
    stages = {"01": stage01, "02": stage02, "03": stage03, "04": stage04, "05": stage05, "06": stage06}
    for p in paths:
        cfg = io.load_config(p, a.default or (Path(__file__).resolve().parents[1] / "configs" / "default.yaml"))
        print(f"===== {cfg['sample_id']} =====", flush=True)
        run_stages = a.stages or (["01", "02", "03", "04", "05"]
                                  + (["06"] if cfg.get("figures", {}).get("save_fullres_previews", False) else []))
        for s in run_stages:
            t0 = time.time(); stages[s](cfg); print(f"[{cfg['sample_id']}] stage {s} {time.time() - t0:.0f}s", flush=True)
