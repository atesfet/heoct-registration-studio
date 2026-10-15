"""Final per-sample deliverables: registered OCT / H&E / common mask with full metadata (TIFF-embedded + JSON)."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import tifffile

from . import io
from . import register as R
from .device import device_name


def sha256(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def write_calibrated_tif(path: Path, arr: np.ndarray, px_um: float, meta_json: str, extra: dict) -> None:
    """ImageJ-calibrated TIFF (unit=um, pixel size) with the full JSON metadata embedded.

    - ImageJ 'Info' property (Fiji: Image > Show Info) holds the complete sample metadata JSON
    - resolution tags carry pixels/um for any TIFF reader
    - private tag 65000 (ASCII) duplicates the JSON for non-ImageJ readers
    """
    md = {"unit": "um", "Info": meta_json, **{k: str(v) for k, v in extra.items()}}
    tifffile.imwrite(path, arr, imagej=True, resolution=(1.0 / px_um, 1.0 / px_um), metadata=md,
                     extratags=[(65000, "s", 0, meta_json, True)], compression="zlib")


def run(cfg: dict) -> None:
    root = Path(cfg["output_root"]) / cfg["sample_id"]
    out = root / "output"; out.mkdir(parents=True, exist_ok=True)
    log = io.get_logger(out, "05_output")
    d1, d2, d3, d4 = (io.stage_dir(cfg, s) for s in ("01_he_preproc", "02_oct_preproc", "03_registration", "04_qc"))
    sid = cfg["sample_id"]
    T = R.Transform.load(d3 / "transform_full.npz")
    oct_px = T.fixed_px_um
    A, _ = T.total_affine()
    sv = 1.0 / np.linalg.svd(A, compute_uv=False)
    he_px = float(np.sqrt(sv[0] * sv[1])) if cfg["output"]["he_native_px_um"] == "auto" else float(cfg["output"]["he_native_px_um"])
    he_px = round(he_px, 4)
    interp = cfg["output"]["interpolation"]

    oct_raw, oct_meta = io.load_oct(cfg["oct_path"])
    oct_clean = tifffile.imread(d2 / "oct_clean_db.tif")
    oct_clean = np.where(tifffile.imread(d3 / "oct_tissue_mask_refined.tif") > 0, oct_clean, np.nan).astype(np.float32)
    oct_mask = tifffile.imread(d3 / "oct_tissue_mask_refined.tif") > 0
    he_norm = tifffile.imread(d1 / "he_stainnorm_rgb.tif")
    he_bg = tifffile.imread(d1 / "he_bg_removed_rgb.tif")
    he_mask = tifffile.imread(d1 / "he_tissue_mask.tif") > 0

    grids = {"oct_grid": oct_px, "he_native_grid": he_px}
    products, files = {}, {}
    for gname, sp in grids.items():
        shape = T.grid_shape_exact(sp)
        log.info(f"{gname}: {sp} um/px, shape {shape}")
        hn = R.warp_image(he_norm, T, True, mode=interp, fill=255.0, spacing_um=sp)
        hb = R.warp_image(he_bg, T, True, mode=interp, fill=255.0, spacing_um=sp)
        hm = R.warp_image(he_mask.astype(np.float32), T, True, mode="bilinear", spacing_um=sp) > 0.5
        if gname == "oct_grid":
            oc, om = oct_clean, oct_mask
        else:
            oc = R.resample_fixed(oct_clean, oct_px, sp, shape, mode=interp)
            om = R.resample_fixed(oct_mask.astype(np.float32), oct_px, sp, shape) > 0.5
        common = om & hm
        products[gname] = {
            f"{sid}_OCT_dB_{gname}.tif": (oc.astype(np.float32), "OCT en-face (top-k) intensity in dB above noise floor; tile flat-field corrected; NaN = background/debris removed"),
            f"{sid}_HE_registered_stainnorm_{gname}.tif": (hn, "H&E, background removed (white) + Macenko stain-normalised, deformably registered to OCT"),
            f"{sid}_HE_registered_original_{gname}.tif": (hb, "H&E, background removed (white), original colours, deformably registered to OCT"),
            f"{sid}_mask_common_{gname}.tif": ((common * 255).astype(np.uint8), "common tissue mask = OCT tissue mask AND registered H&E tissue mask (255 = tissue)"),
            f"{sid}_mask_OCT_{gname}.tif": ((om * 255).astype(np.uint8), "OCT tissue mask"),
            f"{sid}_mask_HE_registered_{gname}.tif": ((hm * 255).astype(np.uint8), "registered H&E tissue mask"),
        }
        for fn, (arr, desc) in products[gname].items():
            files[fn] = {"grid": gname, "pixel_size_um": sp, "shape": list(arr.shape), "dtype": str(arr.dtype), "description": desc}

    qc = json.loads((d4 / "qc_report.json").read_text()) if (d4 / "qc_report.json").exists() else {}
    reg = json.loads((d3 / "registration_info.json").read_text())
    hep = json.loads((d1 / "he_preproc_info.json").read_text())
    ocp = json.loads((d2 / "oct_preproc_info.json").read_text())
    x0, y0 = float(oct_meta.get("oct_x0_mm", 0)), float(oct_meta.get("oct_y0_mm", 0))
    import torch, cv2, skimage
    meta = {
        "sample_id": sid,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": "H&E (moving) deformably registered onto OCT (fixed). All images in this folder share the OCT physical frame.",
        "inputs": {"he": {"path": cfg["he_path"], "sha256": sha256(cfg["he_path"]), "shape": hep["shape"],
                          "pixel_size_um_estimated": he_px, "pixel_size_um_affine_axes": sv.tolist(),
                          "note": "H&E TIFF carries no physical calibration; pixel size estimated from the OCT-calibrated affine fit"},
                   "oct": {"path": cfg["oct_path"], "sha256": sha256(cfg["oct_path"]), "shape": list(oct_raw.shape),
                           "pixel_size_um": oct_px, "value_unit": oct_meta.get("oct_value_unit"),
                           "x0_mm": x0, "y0_mm": y0, "source_metadata": {k: v for k, v in oct_meta.items() if k != "software_json"}}},
        "coordinate_frame": {
            "name": "OCT frame", "units": "um", "origin": "top-left edge of OCT pixel (0,0)",
            "axes": "x = columns (OCT fast scan), y = rows (OCT slow scan)",
            "pixel_centre_um": "x_um = (col + 0.5) * pixel_size_um, y_um = (row + 0.5) * pixel_size_um",
            "to_oct_scanner_mm": f"x_mm = {x0} + (x_um - {oct_px / 2}) / 1000, y_mm = {y0} + (y_um - {oct_px / 2}) / 1000",
            "extent_um": [T.ext_um[1], T.ext_um[0]]},
        "grids": {g: {"pixel_size_um": sp, "shape_rows_cols": list(T.grid_shape_exact(sp)),
                      "note": ("identical to the OCT acquisition grid (OCT at native resolution)" if g == "oct_grid" else
                               "H&E native resolution over the OCT frame; OCT and masks resampled onto it")}
                  for g, sp in grids.items()},
        "files": files,
        "transform": {"definition": R.__doc__, "file": str(d3 / "transform_full.npz"),
                      "dense_map_file": str(d3 / "map_oct_px_to_he_px.npy"),
                      "affine_total_A_hepx_per_um": A.tolist(), "affine_total_t_hepx": T.total_affine()[1].tolist(),
                      "deform_ctrl_spacings_um": [s for s, _ in T.ctrl], "init": reg.get("init"), "affine": reg.get("affine"),
                      "interpolation_output": interp},
        "preprocessing": {"he": {k: hep[k] for k in ("background_rgb", "tissue_fraction", "stain_matrix_he_columns", "max_concentration", "ref_he", "ref_max_c", "params") if k in hep},
                          "oct": {k: ocp[k] for k in ("threshold_db", "clip_db", "mask_fraction", "removed_debris_fraction", "flatfield_pattern_range_db", "params") if k in ocp}},
        "registration_params": cfg["registration"],
        "accuracy": {"mask_dice": {k: qc.get(f"mask_dice_{k}") for k in ("affine", "deformable")},
                     "tissue_edge_distance": {k: qc.get(f"boundary_{k}") for k in ("affine", "deformable")},
                     "landmark_tre": {k: qc.get(f"landmark_{k}") for k in ("affine", "deformable")},
                     "residual_block_matching": {k: qc.get(f"residual_{k}") for k in ("affine", "deformable")}},
        "software": {"python": platform.python_version(), "torch": torch.__version__, "opencv": cv2.__version__,
                     "scikit-image": skimage.__version__, "tifffile": tifffile.__version__, "numpy": np.__version__,
                     "compute_device": device_name(),
                     "code": str(Path(__file__).resolve().parents[1]), "config": cfg.get("_config_path"),
                     "default_config": cfg.get("_default_config_path")},
    }
    mjson = json.dumps(meta, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o))
    (out / f"{sid}_metadata.json").write_text(mjson)
    for gname, prods in products.items():
        for fn, (arr, desc) in prods.items():
            write_calibrated_tif(out / fn, arr, grids[gname], mjson,
                                 {"sample_id": sid, "grid": gname, "pixel_size_um": grids[gname], "content": desc})
            log.info(f"wrote {fn} {arr.shape} {arr.dtype}")
    log.info(f"metadata -> {out / f'{sid}_metadata.json'}")
