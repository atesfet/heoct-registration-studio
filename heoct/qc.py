"""QC for H&E -> OCT registration: overlays, checkerboards, zoomed ROIs, residual-shift map, HTML report."""
from __future__ import annotations

import html
import shutil
import json

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .device import DEVICE  # noqa: E402

from . import io  # noqa: E402
from .features import mind  # noqa: E402
from .register import Transform  # noqa: E402

OCT_CMAP = "gray"


# ------------------------------------------------------------ image composition helpers
def oct_rgb(o: np.ndarray) -> np.ndarray:
    g = (np.clip(o, 0, 1) * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, -1)


def blend(he: np.ndarray, o: np.ndarray, a: float = 0.5) -> np.ndarray:
    return (a * he.astype(np.float32) + (1 - a) * oct_rgb(o).astype(np.float32)).astype(np.uint8)


def false_color(he_int: np.ndarray, o: np.ndarray) -> np.ndarray:
    """OCT -> magenta, H&E optical density -> green; aligned structures appear white/grey."""
    g = np.clip(he_int, 0, 1); m = np.clip(o, 0, 1)
    return (np.stack([m, g, m], -1) * 255).astype(np.uint8)


def checkerboard(he: np.ndarray, o: np.ndarray, tile_px: int, invert: bool = False) -> np.ndarray:
    """Checkerboard A (H&E on even tiles) or its complement B (invert=True); A+B show every pixel of both."""
    yy, xx = np.indices(o.shape)
    sel = ((yy // tile_px + xx // tile_px) % 2 == (1 if invert else 0))[..., None]
    return np.where(sel, he, oct_rgb(o))


def edges(img: np.ndarray, lo=70, hi=160, blur=2.5) -> np.ndarray:
    u = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    u = cv2.GaussianBlur(u, (0, 0), blur)
    return cv2.Canny(u, lo, hi) > 0


def edge_overlay(he_int, o) -> np.ndarray:
    """OCT grey background with OCT edges (magenta) and H&E edges (green)."""
    base = (oct_rgb(o) * 0.6).astype(np.uint8)
    eo, eh = edges(o), edges(he_int)
    base[eo] = (255, 0, 255); base[eh] = (0, 255, 0); base[eo & eh] = (255, 255, 255)
    return base


def contour_overlay(img_rgb, mask_a, mask_b, th=2):
    out = img_rgb.copy()
    for m, col in ((mask_a, (255, 0, 255)), (mask_b, (0, 200, 0))):
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, cs, -1, col, th)
    return out


def boundary_distance(mask_a, mask_b, px_um, frame_margin_px=10):
    """Edge-to-edge distance (um) between two tissue masks, both directions, ignoring edges on the image frame.

    Directly measures rim (epidermis) alignment; mask Dice is insensitive to a thin rim offset.
    """
    from scipy import ndimage as ndi
    H, W = mask_a.shape
    inner = np.zeros_like(mask_a, bool); inner[frame_margin_px:H - frame_margin_px, frame_margin_px:W - frame_margin_px] = True
    def edge(m):
        return m & ~(cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & inner
    ea, eb = edge(mask_a), edge(mask_b)
    if not ea.any() or not eb.any():
        return {}
    da = ndi.distance_transform_edt(~eb)[ea] * px_um     # OCT edge -> nearest H&E edge
    db = ndi.distance_transform_edt(~ea)[eb] * px_um
    d = np.concatenate([da, db])
    return {"median_um": float(np.median(d)), "mean_um": float(d.mean()), "p90_um": float(np.percentile(d, 90)),
            "frac_below_10um": float((d < 10).mean()), "oct_to_he_median_um": float(np.median(da)), "he_to_oct_median_um": float(np.median(db))}


def ds(img, f):
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


# ------------------------------------------------------------ residual-shift estimation
@torch.no_grad()
def residual_shifts(o, he_int, mask, px_um, patch_um, search_um, level_um=4.0, device=DEVICE):
    """Patchwise translation (um) that best re-aligns warped H&E to OCT (MIND-SSD block matching).

    Returned vectors are the remaining local misalignment; ~0 everywhere means good registration.
    """
    f = level_um / px_um
    h, w = int(round(o.shape[0] / f)), int(round(o.shape[1] / f))
    ot = torch.tensor(cv2.resize(o, (w, h), interpolation=cv2.INTER_AREA), device=device)[None, None]
    ht = torch.tensor(cv2.resize(he_int, (w, h), interpolation=cv2.INTER_AREA), device=device)[None, None]
    mt = torch.tensor(cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA), device=device)[None, None]
    Mf, Mm = mind(ot), mind(ht)
    P = int(round(patch_um / level_um)); S = int(round(search_um / level_um)); st = P // 2
    pad = F.pad(Mm, (S, S, S, S), mode="replicate")
    costs = []
    for dy in range(-S, S + 1):
        for dx in range(-S, S + 1):
            sh = pad[..., S + dy:S + dy + h, S + dx:S + dx + w]
            costs.append(F.avg_pool2d(((Mf - sh) ** 2).mean(1, keepdim=True), P, st))
    C = torch.cat(costs, 1)[0]                       # (n_shift, py, px)
    cov = F.avg_pool2d(mt, P, st)[0, 0]
    n = 2 * S + 1
    Cg = C.view(n, n, *C.shape[1:])
    idx = torch.argmin(C, 0)
    iy, ix = idx // n, idx % n

    def sub(c_m, c_0, c_p):
        den = c_m - 2 * c_0 + c_p
        return torch.where(den > 1e-9, 0.5 * (c_m - c_p) / den, torch.zeros_like(den)).clamp(-0.5, 0.5)

    gy, gx = torch.meshgrid(torch.arange(C.shape[1], device=device), torch.arange(C.shape[2], device=device), indexing="ij")
    iyc, ixc = iy.clamp(1, n - 2), ix.clamp(1, n - 2)
    oy = sub(Cg[iyc - 1, ixc, gy, gx], Cg[iyc, ixc, gy, gx], Cg[iyc + 1, ixc, gy, gx])
    ox = sub(Cg[iyc, ixc - 1, gy, gx], Cg[iyc, ixc, gy, gx], Cg[iyc, ixc + 1, gy, gx])
    dy = (iy - S).float() + oy; dx = (ix - S).float() + ox
    # match distinctiveness: how much deeper the minimum is than the cost at the patch's own position
    c0 = Cg[S, S]; cmin = C.min(0).values; cmean = C.mean(0)
    distinct = (cmean - cmin) / cmean.clamp(min=1e-9)
    at_border = (iy == 0) | (iy == n - 1) | (ix == 0) | (ix == n - 1)
    valid = (cov > 0.9) & ~at_border & (distinct > 0.05)
    cy = (gy.float() * st + P / 2) * level_um; cx = (gx.float() * st + P / 2) * level_um
    out = {k: v.cpu().numpy() for k, v in dict(dx_um=dx * level_um, dy_um=dy * level_um, valid=valid, cx_um=cx, cy_um=cy,
                                               distinct=distinct, gain=(c0 - cmin) / c0.clamp(min=1e-9),
                                               border=at_border & (cov > 0.9)).items()}
    mag = np.hypot(out["dx_um"], out["dy_um"])[out["valid"]]
    stats = {"n_patches": int(out["valid"].sum()), "n_border_hits": int(out["border"].sum()),
             "median_um": float(np.median(mag)) if mag.size else None,
             "p90_um": float(np.percentile(mag, 90)) if mag.size else None,
             "mean_um": float(mag.mean()) if mag.size else None,
             "frac_below_2um": float((mag < 2).mean()) if mag.size else None,
             "frac_below_4um": float((mag < 4).mean()) if mag.size else None,
             "patch_um": patch_um, "search_um": search_um, "level_um": level_um}
    return out, stats


def plot_landmarks(pairs, st, bg, px_um, path, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={"width_ratios": [1.4, 1]})
    ext = (0, bg.shape[1] * px_um * 8, bg.shape[0] * px_um * 8, 0)
    axs[0].imshow(bg, cmap="gray", extent=ext)
    yx = pairs["oct_yx_px"] * px_um
    sc = axs[0].scatter(yx[:, 1], yx[:, 0], c=pairs["err_um"], cmap="turbo", vmin=0, vmax=10, s=6)
    plt.colorbar(sc, ax=axs[0], fraction=0.03, label="landmark error (um)")
    axs[0].set_title(title, fontsize=9)
    axs[1].hist(pairs["err_um"], bins=np.arange(0, 21, 0.5), color="0.4")
    for q, c in ((st["median_um"], "r"), (st["p90_um"], "b")):
        axs[1].axvline(q, color=c, ls="--")
    axs[1].set_xlabel("centroid distance (um)"); axs[1].set_ylabel("matched lumens")
    axs[1].set_title(f"n={st['n_matched']}  median {st['median_um']:.2f} um  p90 {st['p90_um']:.2f} um", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_residuals(res, stats, bg, px_um, path, title):
    fig, ax = plt.subplots(figsize=(7, 10))
    ax.imshow(bg, cmap="gray", extent=(0, bg.shape[1] * px_um, bg.shape[0] * px_um, 0))
    v = res["valid"]; mag = np.hypot(res["dx_um"], res["dy_um"])
    sc = ax.scatter(res["cx_um"][v], res["cy_um"][v], c=mag[v], cmap="turbo", vmin=0, vmax=10, s=18)
    ax.quiver(res["cx_um"][v], res["cy_um"][v], res["dx_um"][v], res["dy_um"][v], angles="xy",
              scale_units="xy", scale=0.1, color="w", width=0.002)
    if res["border"].any():
        ax.scatter(res["cx_um"][res["border"]], res["cy_um"][res["border"]], marker="x", c="r", s=14, label="search border hit")
        ax.legend(loc="lower right", fontsize=7)
    plt.colorbar(sc, ax=ax, fraction=0.03, label="residual |shift| (um)")
    ax.set_title(f"{title}\nmedian {stats['median_um']:.2f} um, p90 {stats['p90_um']:.2f} um "
                 f"(n={stats['n_patches']}; arrows x10)", fontsize=9)
    ax.set_xlabel("x (um, OCT)"); ax.set_ylabel("y (um, OCT)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------ ROI selection & panels
def pick_rois(o, he_int, mask, px_um, n, roi_um, manual_um):
    r = int(roi_um / px_um)
    er = cv2.erode(mask.astype(np.uint8), np.ones((r, r), np.uint8)) > 0
    # structure score: product of local std in both modalities (coarse grid)
    k = r // 2
    def lstd(x):
        m = cv2.blur(x, (k, k)); return np.sqrt(np.clip(cv2.blur(x * x, (k, k)) - m * m, 0, None))
    score = lstd(o) * lstd(he_int) * er
    rois = [("manual", int(x / px_um), int(y / px_um)) for x, y in manual_um]
    # always include one ROI on the epidermis / tissue edge (top, left, right, bottom-most boundary points)
    ys, xs = np.nonzero(mask)
    s2 = score.copy()
    taken = np.zeros_like(mask, bool)
    for _ in range(n):
        yx = np.unravel_index(np.argmax(s2), s2.shape)
        if s2[yx] <= 0:
            break
        rois.append(("auto", int(yx[1]), int(yx[0])))
        y0, x0 = yx
        s2[max(0, y0 - 2 * r):y0 + 2 * r, max(0, x0 - 2 * r):x0 + 2 * r] = 0
    # edge ROIs: centre on the tissue boundary at 4 compass points (captures epidermis alignment)
    bd = mask & ~(cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0)
    by, bx = np.nonzero(bd)
    H, W = mask.shape
    inside = (by > r // 2) & (by < H - r // 2) & (bx > r // 2) & (bx < W - r // 2)
    by, bx = by[inside], bx[inside]
    cy, cx = ys.mean(), xs.mean()
    for ang in (90, 210, 330):
        a = np.deg2rad(ang)
        proj = (bx - cx) * np.cos(a) - (by - cy) * np.sin(a)
        i = np.argmax(proj)
        inward = np.array([cx - bx[i], cy - by[i]], float); inward /= np.linalg.norm(inward) + 1e-9
        ex, ey = bx[i] + inward[0] * 0.35 * r, by[i] + inward[1] * 0.35 * r
        ex = int(np.clip(ex, r // 2, W - r // 2 - 1)); ey = int(np.clip(ey, r // 2, H - r // 2 - 1))
        rois.append(("edge", ex, ey))
    return rois, r


def draw_grid(ax, size_px, px_um, step_um=100):
    step = step_um / px_um
    for t in np.arange(step, size_px, step):
        ax.axhline(t, color="yellow", lw=0.4, alpha=0.55); ax.axvline(t, color="yellow", lw=0.4, alpha=0.55)
    c = size_px / 2
    ax.plot([c - 8, c + 8], [c, c], color="cyan", lw=0.8); ax.plot([c, c], [c - 8, c + 8], color="cyan", lw=0.8)


def roi_panel(i, kind, cx, cy, r, px_um, data, path):
    y0, x0 = max(0, cy - r // 2), max(0, cx - r // 2)
    sl = np.s_[y0:y0 + r, x0:x0 + r]
    o = data["oct"][sl]
    rows = [("affine", data["he_aff"][sl], data["int_aff"][sl]), ("deformable", data["he_def"][sl], data["int_def"][sl])]
    cols = ["H&E (warped to OCT)", "OCT", "blend 50/50", "checkerboard A (100 um)", "checkerboard B (complement)",
            "false colour (OCT=magenta, H&E=green)", "edges (OCT=magenta, H&E=green)"]
    fig, axs = plt.subplots(2, 7, figsize=(28, 8.6))
    tp = int(100 / px_um)
    for ri, (tag, he, hi) in enumerate(rows):
        ims = [he, oct_rgb(o), blend(he, o), checkerboard(he, o, tp), checkerboard(he, o, tp, True), false_color(hi, o), edge_overlay(hi, o)]
        for ci, im in enumerate(ims):
            ax = axs[ri, ci]; ax.imshow(im, interpolation="nearest"); draw_grid(ax, im.shape[0], px_um)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0: ax.set_title(cols[ci], fontsize=10)
            if ci == 0: ax.set_ylabel(tag, fontsize=12)
    fig.suptitle(f"ROI {i:02d} ({kind}) centre x={cx * px_um:.0f} um, y={cy * px_um:.0f} um (OCT px {cx},{cy}); "
                 f"{r * px_um:.0f} um field, yellow grid = 100 um, identical pixel coordinates in every panel", fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def roi_locator(o, rois, r, path):
    im = oct_rgb(o).copy()
    for i, (kind, cx, cy) in enumerate(rois):
        col = {"auto": (255, 200, 0), "edge": (0, 255, 255), "manual": (255, 0, 0)}[kind]
        cv2.rectangle(im, (cx - r // 2, cy - r // 2), (cx + r // 2, cy + r // 2), col, 8)
        cv2.putText(im, f"{i:02d}", (cx - r // 2 + 10, cy - r // 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 3, col, 8)
    io.save_png(path, im, 1600)


def deformation_figure(T: Transform, mask, px_um, path):
    X = T.grid_um(16.0)
    with torch.no_grad():
        u = T.displacement(X).cpu().numpy()
    mag = np.hypot(u[..., 0], u[..., 1])
    m = cv2.resize(mask.astype(np.uint8), (mag.shape[1], mag.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    fig, axs = plt.subplots(1, 2, figsize=(12, 9))
    ext = (0, T.ext_um[1], T.ext_um[0], 0)
    im = axs[0].imshow(np.where(m, mag, np.nan), cmap="viridis", extent=ext)
    plt.colorbar(im, ax=axs[0], fraction=0.035, label="|u| (um)")
    axs[0].set_title("non-rigid displacement magnitude (on top of affine)")
    Xn = X.cpu().numpy(); step = 12
    gx, gy = Xn[..., 0] + u[..., 0] * 10, Xn[..., 1] + u[..., 1] * 10
    for k in range(0, gx.shape[0], step): axs[1].plot(gx[k], gy[k], "k-", lw=0.5)
    for k in range(0, gx.shape[1], step): axs[1].plot(gx[:, k], gy[:, k], "k-", lw=0.5)
    axs[1].set_xlim(0, T.ext_um[1]); axs[1].set_ylim(T.ext_um[0], 0); axs[1].set_aspect("equal")
    axs[1].set_title("deformation grid (displacement x10, 192 um spacing)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return {"max_um": float(np.nanmax(np.where(m, mag, np.nan))), "median_um": float(np.nanmedian(np.where(m, mag, np.nan)))}


# ------------------------------------------------------------ main
def run(cfg: dict) -> None:
    out = io.stage_dir(cfg, "04_qc"); log = io.get_logger(out, "04_qc")
    d2, d3 = io.stage_dir(cfg, "02_oct_preproc"), io.stage_dir(cfg, "03_registration")
    q = cfg["qc"]; px = 2.0
    o = tifffile.imread(d2 / "oct_norm.tif"); om = tifffile.imread(d3 / "oct_tissue_mask_refined.tif") > 0
    he_int_native = tifffile.imread(d3 / "he_registration_intensity.tif")
    T_aff = Transform.load(d3 / "transform_affine.npz"); T_def = Transform.load(d3 / "transform_full.npz")
    from .register import warp_image
    data = {"oct": o,
            "he_aff": tifffile.imread(d3 / "he_stainnorm_warped_affine.tif"),
            "he_def": tifffile.imread(d3 / "he_stainnorm_warped_deformable.tif"),
            "int_aff": warp_image(he_int_native, T_aff, False),
            "int_def": tifffile.imread(d3 / "he_intensity_warped_deformable.tif")}
    hm_aff = tifffile.imread(d3 / "he_mask_warped_affine.tif") > 0
    hm_def = tifffile.imread(d3 / "he_mask_warped_deformable.tif") > 0
    report = {"sample_id": cfg["sample_id"]}

    # 1. overview figures
    ov = out / "overview"; ov.mkdir(exist_ok=True)
    f = 0.25
    for tag in ("affine", "deformable"):
        he, hi, hm = data[f"he_{tag[:3]}"], data[f"int_{tag[:3]}"], (hm_aff if tag == "affine" else hm_def)
        io.save_png(ov / f"{tag}_01_he_warped.png", ds(he, f))
        io.save_png(ov / f"{tag}_02_blend.png", ds(blend(he, o), f))
        io.save_png(ov / f"{tag}_03a_checkerboard_500um_A.png", ds(checkerboard(he, o, int(500 / px)), f))
        io.save_png(ov / f"{tag}_03b_checkerboard_500um_B.png", ds(checkerboard(he, o, int(500 / px), True), f))
        io.save_png(ov / f"{tag}_04_falsecolor.png", ds(false_color(hi, o), f))
        io.save_png(ov / f"{tag}_05_mask_contours.png", ds(contour_overlay(blend(he, o), om, hm, 6), f))
        inter = (om & hm).sum(); report[f"mask_dice_{tag}"] = float(2 * inter / (om.sum() + hm.sum()))
        report[f"boundary_{tag}"] = boundary_distance(om, hm, px)
    io.save_png(ov / "oct_reference.png", ds(oct_rgb(o), f))
    report["deformation"] = deformation_figure(T_def, om, px, ov / "deformation_field.png")
    log.info(f"mask dice affine {report['mask_dice_affine']:.4f} deformable {report['mask_dice_deformable']:.4f}")
    log.info(f"tissue-edge distance affine {report['boundary_affine']}  deformable {report['boundary_deformable']}")

    # 2. residual misalignment maps
    rs = out / "residuals"; rs.mkdir(exist_ok=True)
    bg = ds(o, 0.125)
    for tag in ("affine", "deformable"):
        res, st = residual_shifts(o, data[f"int_{tag[:3]}"], om, px, q["residual_patch_um"], q["residual_search_um"])
        np.savez(rs / f"residual_shifts_{tag}.npz", **res)
        plot_residuals(res, st, bg, px * 8, rs / f"residual_map_{tag}.png", f"local residual shift after {tag}")
        report[f"residual_{tag}"] = st
        log.info(f"residual {tag}: {st}")

    # 2b. independent lumen-landmark TRE
    oct_db_lumen = np.nan_to_num(tifffile.imread(d2 / "oct_clean_db.tif"), nan=0.0).astype(np.float32)
    for tag in ("affine", "deformable"):
        # OCT lumens = flat-fielded signal <= qc.oct_lumen_db dB above the noise floor: an absolute, physical threshold,
        # identical for every sample and independent of the display/registration standardisation
        pairs, st = landmark_tre(oct_db_lumen, data[f"int_{tag[:3]}"], om, px, oct_thr=q.get("oct_lumen_db", 2.6))
        st["oct_lumen_threshold_db"] = q.get("oct_lumen_db", 2.6)
        report[f"landmark_{tag}"] = st
        np.savez(rs / f"landmarks_{tag}.npz", **pairs)
        log.info(f"landmark TRE {tag}: {st}")
        if pairs:
            plot_landmarks(pairs, st, bg, px, rs / f"landmark_tre_{tag}.png", f"lumen-landmark TRE after {tag}")
    init = json.loads((d3 / "registration_info.json").read_text())["init"]
    report["orientation"] = {"mirrored": init["best_flip"] < 0, "rotation_deg": init["best_rot_deg"],
                             "he_px_um_init": init["best_he_px_um"]}

    # 3. zoomed ROI panels
    rd = out / "rois"; shutil.rmtree(rd, ignore_errors=True); rd.mkdir()
    rois, r = pick_rois(o, data["int_def"], om, px, q["n_auto_rois"], q["roi_size_um"], q["manual_rois_um"])
    roi_locator(o, rois, r, rd / "roi_locator.png")
    report["rois"] = []
    for i, (kind, cx, cy) in enumerate(rois):
        p = rd / f"roi_{i:02d}_{kind}_x{cx * px:.0f}um_y{cy * px:.0f}um.png"
        roi_panel(i, kind, cx, cy, r, px, data, p)
        report["rois"].append({"id": i, "kind": kind, "cx_um": cx * px, "cy_um": cy * px, "file": p.name})
    io.write_json(out / "qc_report.json", report)
    write_html(out, report)
    log.info(f"QC written to {out}")


def write_html(out, rep):
    def img(p, w="100%"):
        return f'<a href="{p}"><img src="{p}" style="width:{w}"></a>'
    ra, rd_ = rep["residual_affine"], rep["residual_deformable"]
    rows = "".join(f"<tr><td>{k}</td><td>{rep[f'mask_dice_{k}']:.4f}</td><td>{rep[f'landmark_{k}'].get('median_um', float('nan')):.2f}</td>"
                   f"<td>{rep[f'landmark_{k}'].get('p90_um', float('nan')):.2f}</td><td>{rep[f'landmark_{k}'].get('n_matched', 0)}</td>"
                   f"<td>{rep[f'residual_{k}']['median_um']:.2f}</td>"
                   f"<td>{rep[f'residual_{k}']['p90_um']:.2f}</td><td>{100 * rep[f'residual_{k}']['frac_below_4um']:.0f}%</td>"
                   f"<td>{rep[f'residual_{k}']['n_patches']}</td></tr>" for k in ("affine", "deformable"))
    ori = rep["orientation"]
    rois = "".join(f"<h3>ROI {r['id']:02d} ({r['kind']}) x={r['cx_um']:.0f} um y={r['cy_um']:.0f} um</h3>{img('rois/' + r['file'])}" for r in rep["rois"])
    ov = "".join(f"<div class=c><p>{t}</p>{img('overview/' + t)}</div>" for t in
                 ["oct_reference.png"] + [f"{k}_{s}" for k in ("affine", "deformable") for s in
                                          ("01_he_warped.png", "02_blend.png", "03a_checkerboard_500um_A.png", "03b_checkerboard_500um_B.png", "04_falsecolor.png", "05_mask_contours.png")])
    doc = f"""<!doctype html><html><head><meta charset=utf-8><title>QC {html.escape(rep['sample_id'])}</title>
<style>body{{font-family:sans-serif;margin:20px;background:#fafafa;color:#222}} .c{{display:inline-block;width:32%;vertical-align:top;margin:4px}}
table{{border-collapse:collapse}} td,th{{border:1px solid #999;padding:4px 10px}}</style></head><body>
<h1>H&amp;E &rarr; OCT registration QC: {html.escape(rep['sample_id'])}</h1>
<p>Fixed = OCT (2 um/px grid). Moving = stain-normalised H&amp;E. All images below are on the OCT pixel grid.</p>
<p>Orientation found: mirrored = <b>{ori['mirrored']}</b>, rotation {ori['rotation_deg']:.1f} deg, H&amp;E pixel ~{ori['he_px_um_init']:.3f} um.</p>
<table><tr><th>stage</th><th>mask Dice</th><th>landmark TRE median (um)</th><th>landmark TRE p90 (um)</th><th>n landmarks</th><th>residual median (um)</th><th>residual p90 (um)</th><th>patches &lt;4 um</th><th>n patches</th></tr>{rows}</table>
<p>Residual = local translation (MIND block matching, {ra['patch_um']} um patches, +/-{ra['search_um']} um search at {ra['level_um']} um/px)
that would further align warped H&amp;E to OCT. It uses the same similarity measure as the optimiser, so treat it as an upper bound on
consistency rather than ground-truth TRE; the zoomed ROIs are the visual check.</p>
<p>Non-rigid displacement: median {rep['deformation']['median_um']:.1f} um, max {rep['deformation']['max_um']:.1f} um.</p>
<h2>Independent lumen-landmark TRE</h2><p>Small unstained/low-scattering lumens segmented independently in OCT and warped H&amp;E; mutual nearest neighbours (&lt;30 um, similar size) are matched and centroid distance reported.</p>
<div class=c>{img('residuals/landmark_tre_affine.png')}</div><div class=c>{img('residuals/landmark_tre_deformable.png')}</div>
<h2>Residual maps</h2><div class=c>{img('residuals/residual_map_affine.png')}</div><div class=c>{img('residuals/residual_map_deformable.png')}</div>
<div class=c>{img('overview/deformation_field.png')}</div>
<h2>Overview</h2>{ov}
<h2>Zoomed ROIs</h2><div class=c style="width:40%">{img('rois/roi_locator.png')}</div>{rois}
</body></html>"""
    (out / "index.html").write_text(doc)


# ------------------------------------------------------------ independent landmark check
def lumen_landmarks(img, mask, thr=0.1, erode_px=41, area_px=(15, 3000), return_mask=False):
    """Centroids (y,x px) and areas of small dark lumens/gaps (vessels, clefts) inside tissue.

    In OCT these are low-scattering holes; in H&E optical density they are unstained gaps,
    so both images show them as near-zero intensity -> modality-independent landmarks.
    """
    from scipy import ndimage as ndi
    er = cv2.erode(mask.astype(np.uint8), np.ones((erode_px, erode_px), np.uint8)) > 0
    b = (cv2.GaussianBlur(img, (0, 0), 1.0) < thr) & er
    b = cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lab, n = ndi.label(b)
    if n == 0:
        return (np.zeros((0, 2)), np.zeros(0), np.zeros_like(mask, bool)) if return_mask else (np.zeros((0, 2)), np.zeros(0))
    idx = np.arange(1, n + 1)
    a = ndi.sum(b, lab, idx); c = np.array(ndi.center_of_mass(b, lab, idx))
    k = (a >= area_px[0]) & (a <= area_px[1])
    if return_mask:   # binary mask of the kept lumens (for figures)
        return c[k], a[k], np.isin(lab, idx[k])
    return c[k], a[k]


def landmark_tre(o, he_int_warped, mask, px_um, max_dist_px=10, min_size_ratio=0.3, ratio=0.5, oct_thr=0.1):
    """Lumen-centroid matching -> independent TRE (um).

    A pair (OCT i, H&E j) is accepted if i and j are mutual nearest neighbours within max_dist_px,
    have similar areas, and the match is unambiguous in both directions (Lowe ratio test:
    nearest / second-nearest distance < ratio), which rejects chance pairings in lumen-dense regions.
    """
    from scipy.spatial import cKDTree
    co, ao = lumen_landmarks(o, mask, thr=oct_thr); ch, ah = lumen_landmarks(he_int_warped, mask)
    if len(co) < 2 or len(ch) < 2:
        return {}, {"n_matched": 0}
    th, to = cKDTree(ch), cKDTree(co)
    d, j = th.query(co, k=2)
    db, ib = to.query(ch, k=2)
    i = np.arange(len(co)); j1 = j[:, 0]
    ok = (d[:, 0] <= max_dist_px) & (ib[j1, 0] == i)
    ok &= d[:, 0] < ratio * d[:, 1]                      # unambiguous in H&E
    ok &= db[j1, 0] < ratio * db[j1, 1]                  # unambiguous in OCT
    sr = np.minimum(ao, ah[j1]) / np.maximum(ao, ah[j1])
    sel = np.flatnonzero(ok & (sr > min_size_ratio))
    err = d[sel, 0] * px_um
    if err.size == 0:
        return {}, {"n_matched": 0}
    pairs = {"oct_yx_px": co[sel], "he_yx_px": ch[j1[sel]], "err_um": err}
    return pairs, {"n_oct": int(len(co)), "n_he": int(len(ch)), "n_matched": int(len(sel)),
                   "median_um": float(np.median(err)), "mean_um": float(err.mean()),
                   "p75_um": float(np.percentile(err, 75)), "p90_um": float(np.percentile(err, 90)),
                   "frac_below_2um": float((err < 2).mean()), "frac_below_4um": float((err < 4).mean()),
                   "match_rule": f"mutual NN <= {max_dist_px * px_um:g} um, area ratio > {min_size_ratio}, Lowe ratio < {ratio}"}
