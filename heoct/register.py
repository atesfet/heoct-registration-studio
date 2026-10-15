"""GPU multimodal 2D registration: H&E (moving) -> OCT (fixed).

Transform chain (fixed OCT physical coords x [um] -> H&E native pixel coords p):
    y = x + u(x)                      u: sum of cubic B-spline-like control grids [um]
    z = (I + M)(y - c) + c + d        affine refinement around fixed-image centre c
    p = A0 z + t0                     initial similarity (scale, rotation, flip) [px/um]
All registration outputs live on the OCT pixel grid (pixel j centre at (j+0.5)*px_um).
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .device import DEVICE, empty_cache

from .features import gblur, mind


# ------------------------------------------------------------------ separable bicubic interpolation
def _cubic_weight_matrix(coord_norm: torch.Tensor, size: int) -> torch.Tensor:
    """(n, size) weights reproducing F.grid_sample(mode="bicubic", align_corners=True, padding_mode="border").

    Cubic convolution with A = -0.75 (as in PyTorch); neighbour indices are clamped to the border.
    For a regular (meshgrid) sample grid, bicubic sampling of a control grid C is then Wy @ C @ Wx^T: two small
    matrix products instead of a per-pixel gather, ~exact and fast on CPU and Apple-silicon (MPS) GPUs.
    """
    A = -0.75
    x = (coord_norm + 1) / 2 * (size - 1)
    f = torch.floor(x); t = x - f
    def w_near(d):                     # |d| <= 1
        return ((A + 2) * d - (A + 3)) * d * d + 1
    def w_far(d):                      # 1 < |d| < 2
        return ((A * d - 5 * A) * d + 8 * A) * d - 4 * A
    ws = [w_far(t + 1), w_near(t), w_near(1 - t), w_far(2 - t)]
    W = torch.zeros(x.shape[0], size, dtype=coord_norm.dtype, device=coord_norm.device)
    rows = torch.arange(x.shape[0], device=coord_norm.device)
    for k, wk in enumerate(ws):
        idx = (f.long() - 1 + k).clamp(0, size - 1)
        W.index_put_((rows, idx), wk, accumulate=True)
    return W


# ------------------------------------------------------------------ transform
class Transform:
    def __init__(self, fixed_shape, fixed_px_um, moving_shape, device=DEVICE):
        self.fixed_shape = tuple(int(v) for v in fixed_shape)          # (H, W) pixels
        self.fixed_px_um = float(fixed_px_um)
        self.moving_shape = tuple(int(v) for v in moving_shape)        # (Hm, Wm) pixels
        self.ext_um = (self.fixed_shape[0] * fixed_px_um, self.fixed_shape[1] * fixed_px_um)  # (Hum, Wum)
        self.device = device
        self.c = torch.tensor([self.ext_um[1] / 2, self.ext_um[0] / 2], device=device)  # (x, y)
        self.A0 = torch.eye(2, device=device)
        self.t0 = torch.zeros(2, device=device)
        self.M = torch.zeros(2, 2, device=device)
        self.d = torch.zeros(2, device=device)
        self.ctrl: list[tuple[float, torch.Tensor]] = []   # (spacing_um, (1,2,gh,gw) displacement um)

    # -- coordinates
    def grid_um(self, spacing_um: float) -> torch.Tensor:
        h = int(round(self.ext_um[0] / spacing_um)); w = int(round(self.ext_um[1] / spacing_um))
        ys = (torch.arange(h, device=self.device) + 0.5) * (self.ext_um[0] / h)
        xs = (torch.arange(w, device=self.device) + 0.5) * (self.ext_um[1] / w)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], -1)   # (h, w, 2) as (x, y)

    def grid_um_exact(self, spacing_um: float, rows=None) -> torch.Tensor:
        """Grid with exact pixel spacing starting at the OCT frame's top-left edge (covers the OCT extent)."""
        h = int(math.ceil(self.ext_um[0] / spacing_um - 1e-6)); w = int(math.ceil(self.ext_um[1] / spacing_um - 1e-6))
        r0, r1 = (0, h) if rows is None else rows
        ys = (torch.arange(r0, min(r1, h), device=self.device) + 0.5) * spacing_um
        xs = (torch.arange(w, device=self.device) + 0.5) * spacing_um
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        return torch.stack([gx, gy], -1)

    def grid_shape_exact(self, spacing_um: float) -> tuple[int, int]:
        return (int(math.ceil(self.ext_um[0] / spacing_um - 1e-6)), int(math.ceil(self.ext_um[1] / spacing_um - 1e-6)))

    def displacement(self, X: torch.Tensor, ctrl_list=None) -> torch.Tensor:
        ctrl_list = self.ctrl if ctrl_list is None else ctrl_list
        if not ctrl_list:
            return torch.zeros_like(X)
        if X.ndim == 3 and X.shape[0] > 1 and X.shape[1] > 1 and torch.equal(X[0, :, 0], X[-1, :, 0]) and torch.equal(X[:, 0, 1], X[:, -1, 1]):
            # regular grid (all registration levels and exports): separable bicubic = two matrix products
            xn = X[0, :, 0] / self.ext_um[1] * 2 - 1
            yn = X[:, 0, 1] / self.ext_um[0] * 2 - 1
            u = 0
            for _, cg in ctrl_list:
                Wy = _cubic_weight_matrix(yn, cg.shape[2]); Wx = _cubic_weight_matrix(xn, cg.shape[3])
                u = u + (Wy @ cg[0] @ Wx.T).permute(1, 2, 0)
            return u
        Xn = torch.stack([X[..., 0] / self.ext_um[1] * 2 - 1, X[..., 1] / self.ext_um[0] * 2 - 1], -1)[None]
        u = 0
        for _, cg in ctrl_list:
            u = u + F.grid_sample(cg, Xn, mode="bicubic", align_corners=True, padding_mode="border")[0].permute(1, 2, 0)
        return u

    def affine_apply(self, Y: torch.Tensor, M=None, d=None, A0=None, t0=None) -> torch.Tensor:
        M = self.M if M is None else M; d = self.d if d is None else d
        A0 = self.A0 if A0 is None else A0; t0 = self.t0 if t0 is None else t0
        I = torch.eye(2, device=Y.device)
        z = (Y - self.c) @ (I + M).T + self.c + d
        return z @ A0.T + t0

    def map_px(self, X: torch.Tensor, with_deform=True, ctrl_list=None) -> torch.Tensor:
        Y = X + self.displacement(X, ctrl_list) if with_deform else X
        return self.affine_apply(Y)

    def px_to_norm(self, P: torch.Tensor) -> torch.Tensor:
        Hm, Wm = self.moving_shape
        return torch.stack([(P[..., 0] + 0.5) / Wm * 2 - 1, (P[..., 1] + 0.5) / Hm * 2 - 1], -1)

    def total_affine(self) -> tuple[np.ndarray, np.ndarray]:
        """p = A x + t for the affine part only (x in fixed um, p in moving px)."""
        I = torch.eye(2, device=self.device)
        A = self.A0 @ (I + self.M)
        t = self.A0 @ (self.c - (I + self.M) @ self.c + self.d) + self.t0
        return A.detach().cpu().numpy(), t.detach().cpu().numpy()

    def he_px_to_oct_um(self, P: np.ndarray, iters: int = 30) -> np.ndarray:
        """Inverse map for points (N,2) in H&E px (x,y) -> OCT um, via fixed-point iteration."""
        A, t = self.total_affine()
        Ainv = np.linalg.inv(A)
        P = torch.tensor(P, dtype=torch.float32, device=self.device)
        Ai = torch.tensor(Ainv, dtype=torch.float32, device=self.device)
        X = (P - torch.tensor(t, device=self.device)) @ Ai.T
        for _ in range(iters):
            X = X - (self.map_px(X[None])[0] - P) @ Ai.T
        return X.cpu().numpy()

    # -- persistence
    def state(self) -> dict:
        A, t = self.total_affine()
        st = {"fixed_shape": np.array(self.fixed_shape), "fixed_px_um": self.fixed_px_um,
              "moving_shape": np.array(self.moving_shape),
              "A0": self.A0.cpu().numpy(), "t0": self.t0.cpu().numpy(), "M": self.M.cpu().numpy(),
              "d": self.d.cpu().numpy(), "c": self.c.cpu().numpy(),
              "affine_total_A": A, "affine_total_t": t,
              "ctrl_spacings_um": np.array([s for s, _ in self.ctrl])}
        for i, (_, cg) in enumerate(self.ctrl):
            st[f"ctrl_{i}"] = cg.detach().cpu().numpy()
        return st

    def save(self, path) -> None:
        np.savez(path, **self.state())

    @classmethod
    def load(cls, path, device=DEVICE) -> "Transform":
        z = np.load(path)
        T = cls(z["fixed_shape"], float(z["fixed_px_um"]), z["moving_shape"], device)
        for k in ("A0", "t0", "M", "d", "c"):
            setattr(T, k, torch.tensor(z[k], dtype=torch.float32, device=device))
        T.ctrl = [(float(s), torch.tensor(z[f"ctrl_{i}"], device=device))
                  for i, s in enumerate(z["ctrl_spacings_um"])]
        return T


# ------------------------------------------------------------------ helpers
def _t(x: np.ndarray, device) -> torch.Tensor:
    return torch.tensor(np.ascontiguousarray(x), dtype=torch.float32, device=device)[None, None]


def soft_dice(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, a.ndim))
    return (2 * (a * b).sum(dims) + 1e-6) / (a.sum(dims) + b.sum(dims) + 1e-6)


def warp(img: torch.Tensor, T: Transform, X: torch.Tensor, with_deform=True, mode="bilinear") -> torch.Tensor:
    """Sample moving image (1,C,h,w; any pyramid level spanning the full H&E extent) at fixed coords X."""
    g = T.px_to_norm(T.map_px(X, with_deform))[None]
    return F.grid_sample(img, g, mode=mode, padding_mode="zeros", align_corners=False)


class Pyramid:
    """Fixed + moving images resampled for a given fixed-grid spacing."""

    def __init__(self, fixed, fixed_mask, fixed_px, moving, moving_mask, moving_px_est, device):
        self.f, self.fm, self.fpx = fixed, fixed_mask.astype(np.float32), fixed_px
        self.m, self.mm, self.mpx = moving, moving_mask.astype(np.float32), moving_px_est
        self.device = device
        self.cache = {}

    def level(self, s: float):
        if s in self.cache:
            return self.cache[s]
        H, W = self.f.shape
        h, w = int(round(H * self.fpx / s)), int(round(W * self.fpx / s))
        f = cv2.resize(self.f, (w, h), interpolation=cv2.INTER_AREA)
        fm = cv2.resize(self.fm, (w, h), interpolation=cv2.INTER_AREA)
        k = max(1.0, s / self.mpx)
        Hm, Wm = self.m.shape
        hm, wm = int(round(Hm / k)), int(round(Wm / k))
        m = cv2.resize(self.m, (wm, hm), interpolation=cv2.INTER_AREA)
        mm = cv2.resize(self.mm, (wm, hm), interpolation=cv2.INTER_AREA)
        out = tuple(_t(a, self.device) for a in (f, fm, m, mm))
        self.cache[s] = out
        return out


# ------------------------------------------------------------------ registration
class Registrar:
    def __init__(self, fixed, fixed_mask, fixed_px_um, moving, moving_mask, cfg, log, device=DEVICE):
        self.cfg, self.log, self.device = cfg, log, device
        self.T = Transform(fixed.shape, fixed_px_um, moving.shape, device)
        self.pyr = Pyramid(fixed, fixed_mask, fixed_px_um, moving, moving_mask, cfg["he_px_um_init"], device)
        self.history: list[dict] = []
        self.mind_kw_deform = dict(radius=float(cfg["mind_radius_px"]), dilation=int(cfg["mind_dilation_px"]))
        self.mind_kw_affine = dict(radius=float(cfg.get("affine_mind_radius_px", cfg["mind_radius_px"])),
                                   dilation=int(cfg.get("affine_mind_dilation_px", cfg["mind_dilation_px"])))
        self.mind_kw = self.mind_kw_affine

    # ---------------- stage 0: global similarity search on masks
    def init_search(self, s: float = 32.0) -> dict:
        cfg = self.cfg
        f, fm, m, mm = self.pyr.level(s)
        X = self.T.grid_um(s)
        fm_b = gblur(fm, 1.5)
        mm_b = gblur(mm, 1.5)
        # mask centroids
        fmn = fm[0, 0]; Xf = (X * fmn[..., None]).sum((0, 1)) / fmn.sum()
        Hm, Wm = self.T.moving_shape
        mmn = mm[0, 0]; hh, ww = mmn.shape
        ys = (torch.arange(hh, device=self.device) + 0.5) * Hm / hh - 0.5
        xs = (torch.arange(ww, device=self.device) + 0.5) * Wm / ww - 0.5
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        Pc = torch.stack([(gx * mmn).sum() / mmn.sum(), (gy * mmn).sum() / mmn.sum()])
        rots = np.arange(cfg["rot_search_deg"][0], cfg["rot_search_deg"][1] + 1e-6, cfg.get("rot_step_deg", 5.0))
        scales = np.geomspace(cfg["he_px_um_search"][0], cfg["he_px_um_search"][1], cfg.get("n_scales", 5))
        flips = [1.0, -1.0] if cfg["allow_flip"] else [1.0]
        cand = torch.tensor([(fl, r, sc) for fl in flips for r in rots for sc in scales], device=self.device, dtype=torch.float32)
        N = cand.shape[0]
        th = (cand[:, 1] * math.pi / 180).clone().requires_grad_(True)
        ls = torch.log(cand[:, 2]).clone().requires_grad_(True)
        tr = torch.zeros(N, 2, device=self.device, requires_grad=True)
        flip = cand[:, 0]
        opt = torch.optim.Adam([{"params": [th], "lr": 0.01}, {"params": [ls], "lr": 0.01},
                                {"params": [tr], "lr": 0.01}])
        Xr = X.reshape(-1, 2)

        def sample(th, ls, tr, idx=None):
            idx = torch.arange(th.shape[0], device=self.device) if idx is None else idx
            cos, sin = torch.cos(th[idx]), torch.sin(th[idx])
            R = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)  # n,2,2
            Fm = torch.diag_embed(torch.stack([flip[idx], torch.ones_like(flip[idx])], -1))
            A = R @ Fm / torch.exp(ls[idx])[:, None, None]
            P = torch.einsum("nij,nkj->nki", A, Xr[None] - Xf - tr[idx][:, None, :] * 500.0) + Pc
            g = self.T.px_to_norm(P).reshape(len(idx), *X.shape)
            w = F.grid_sample(mm_b.expand(len(idx), -1, -1, -1), g, align_corners=False, padding_mode="zeros")
            return w, A, P

        chunks = torch.arange(N, device=self.device).split(128)
        for it in range(200):
            opt.zero_grad()
            for ch in chunks:
                w, _, _ = sample(th, ls, tr, ch)
                (1 - soft_dice(w, fm_b.expand(len(ch), -1, -1, -1))).sum().backward()
            opt.step()
        with torch.no_grad():
            dice = torch.cat([soft_dice((sample(th, ls, tr, ch)[0] > 0.5).float(),
                                        (fm > 0.5).float().expand(len(ch), -1, -1, -1)) for ch in chunks])
            A_all = torch.cat([sample(th, ls, tr, ch)[1] for ch in chunks])
        # distinct candidates (different flip or rotation > 20 deg apart), best mask Dice first
        order = torch.argsort(dice, descending=True).tolist()
        picked = []
        for i in order:
            ang = float(th[i] * 180 / math.pi)
            if all(float(flip[i]) != float(flip[j]) or abs((ang - float(th[j] * 180 / math.pi) + 180) % 360 - 180) > 20 for j in picked):
                picked.append(i)
            if len(picked) >= cfg.get("n_candidates_refine", 1):
                break
        # re-rank by image similarity: short MIND+Dice affine fit from each candidate
        rerank = []
        for i in picked:
            self.T.A0 = A_all[i].detach().clone()
            self.T.t0 = (Pc - A_all[i] @ (Xf + tr[i] * 500.0)).detach().clone()
            self.T.M = torch.zeros(2, 2, device=self.device); self.T.d = torch.zeros(2, device=self.device)
            M = torch.zeros(2, 2, device=self.device, requires_grad=True)
            dp = torch.zeros(2, device=self.device, requires_grad=True)
            o2 = torch.optim.Adam([M, dp], lr=cfg["lr_affine"])
            for _ in range(100):
                o2.zero_grad(); self.T.M, self.T.d = M, dp * 1000.0
                l, _ = self._loss(s, with_deform=False); l.backward(); o2.step()
            with torch.no_grad():
                self.T.M, self.T.d = M.detach(), dp.detach() * 1000.0
                l, met = self._loss(s, with_deform=False)
            rerank.append({"idx": i, "flip": float(flip[i]), "rot_deg": float(th[i] * 180 / math.pi),
                           "he_px_um": float(torch.exp(ls[i])), "mask_dice": float(dice[i]),
                           "loss_after_short_affine": float(l), **met,
                           "A0": self.T.A0.clone(), "t0": self.T.t0.clone()})
            self.log.info(f"  candidate flip={rerank[-1]['flip']:+.0f} rot={rerank[-1]['rot_deg']:.1f} "
                          f"px={rerank[-1]['he_px_um']:.3f} dice={rerank[-1]['mask_dice']:.4f} -> loss {float(l):.4f} {met}")
        best = min(rerank, key=lambda r: r["loss_after_short_affine"])
        self.T.A0, self.T.t0 = best["A0"], best["t0"]
        self.T.M = torch.zeros(2, 2, device=self.device); self.T.d = torch.zeros(2, device=self.device)
        b = best["idx"]
        res = {"best_flip": float(flip[b]), "best_rot_deg": float(th[b] * 180 / math.pi),
               "best_he_px_um": float(torch.exp(ls[b])), "best_mask_dice": float(dice[b]),
               "n_candidates": N, "reranked": [{k: v for k, v in r.items() if k not in ("A0", "t0")} for r in rerank]}
        self.log.info(f"init search: {res}")
        self.history.append({"stage": "init", **res})
        return res

    def set_fixed_mask(self, mask: np.ndarray) -> None:
        """Replace the fixed (OCT) tissue mask used by the Dice term / loss weighting."""
        self.pyr.fm = mask.astype(np.float32)
        self.pyr.cache.clear()
        if hasattr(self, "_fmind"):
            del self._fmind

    # ---------------- losses
    def _loss(self, s, with_deform=True, ctrl_list=None, w_dice=0.5):
        f, fm, m, mm = self.pyr.level(s)
        X = self.T.grid_um(s)
        g = self.T.px_to_norm(self.T.map_px(X, with_deform, ctrl_list))[None]
        wm = F.grid_sample(m, g, align_corners=False, padding_mode="zeros")
        wmm = F.grid_sample(mm, g, align_corners=False, padding_mode="zeros")
        key = (s, tuple(self.mind_kw.values()))
        if not hasattr(self, "_fmind") or self._fmind[0] != key:
            self._fmind = (key, mind(f, **self.mind_kw), gblur(fm, 4.0).clamp(0, 1))
        _, Mf, roi = self._fmind
        Mm = mind(wm, **self.mind_kw)
        w = torch.maximum(roi, gblur(wmm, 4.0)).clamp(0, 1)
        ssd = (((Mf - Mm) ** 2).mean(1, keepdim=True) * w).sum() / w.sum()
        dice = soft_dice(wmm, fm)[0]
        return ssd + w_dice * (1 - dice), {"mind_ssd": float(ssd), "dice": float(dice)}

    # ---------------- stage 1: affine
    def affine(self):
        cfg = self.cfg
        self.mind_kw = self.mind_kw_affine
        M = self.T.M.detach().clone().requires_grad_(True)
        dp = (self.T.d.detach() / 1000.0).clone().requires_grad_(True)
        for s, iters in zip(cfg["affine_levels_um"], cfg["affine_iters"]):
            opt = torch.optim.Adam([M, dp], lr=cfg["lr_affine"])
            t0 = time.time()
            for it in range(iters):
                opt.zero_grad()
                self.T.M, self.T.d = M, dp * 1000.0
                loss, met = self._loss(s, with_deform=False)
                loss.backward(); opt.step()
                if (it + 1) % max(1, iters // 3) == 0 and it + 1 < iters:
                    self.log.info(f"  affine {s}um: iter {it + 1}/{iters} ({time.time() - t0:.0f}s)")
            self.T.M, self.T.d = M.detach().clone(), (dp * 1000.0).detach().clone()
            with torch.no_grad():
                _, met = self._loss(s, with_deform=False)
            self.log.info(f"affine level {s}um: {met} ({time.time() - t0:.1f}s)")
            self.history.append({"stage": "affine", "level_um": s, **met})
        A, t = self.T.total_affine()
        sv = np.linalg.svd(A, compute_uv=False)
        res = {"A_px_per_um": A, "t_px": t, "he_px_um_from_affine": (1 / sv).tolist(),
               "rotation_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), "det_sign": float(np.sign(np.linalg.det(A)))}
        self.log.info(f"affine result: {res}")
        return res

    # ---------------- stage 2: deformable
    def deform(self):
        cfg = self.cfg
        self.mind_kw = self.mind_kw_deform
        for s, sp, iters, lam in zip(cfg["deform_levels_um"], cfg["deform_ctrl_spacing_um"],
                                     cfg["deform_iters"], cfg["deform_lambda"]):
            gh = int(math.ceil(self.T.ext_um[0] / sp)) + 1
            gw = int(math.ceil(self.T.ext_um[1] / sp)) + 1
            cg = torch.zeros(1, 2, gh, gw, device=self.device, requires_grad=True)
            opt = torch.optim.Adam([cg], lr=cfg["lr_deform"] * sp / 100.0)
            frozen = [(a, b.detach()) for a, b in self.T.ctrl]
            t0 = time.time()
            X = self.T.grid_um(s)
            for it in range(iters):
                opt.zero_grad()
                ctrl = frozen + [(sp, cg)]
                loss, met = self._loss(s, True, ctrl)
                u = self.T.displacement(X, ctrl).permute(2, 0, 1)[None]
                reg, fold = self.regulariser(u, s)
                (loss + lam * reg + 10.0 * fold).backward()
                opt.step()
                if (it + 1) % max(1, iters // 5) == 0 and it + 1 < iters:
                    self.log.info(f"  deform {s}um ctrl {sp}um: iter {it + 1}/{iters} loss {float(loss):.4f} ({time.time() - t0:.0f}s)")
            self.T.ctrl = frozen + [(sp, cg.detach().clone())]
            with torch.no_grad():
                _, met = self._loss(s, True)
                u = self.T.displacement(X).permute(2, 0, 1)[None]
                reg, fold = self.regulariser(u, s)
                jac = self.jacobian_det(u, s)
            met |= {"reg": float(reg), "max_disp_um": float(u.norm(dim=1).max()),
                    "jac_min": float(jac.min()), "jac_max": float(jac.max()), "fold_frac": float((jac <= 0).float().mean())}
            self.log.info(f"deform level {s}um ctrl {sp}um: {met} ({time.time() - t0:.1f}s)")
            self.history.append({"stage": "deform", "level_um": s, "ctrl_um": sp, **met})

    @staticmethod
    def jacobian_det(u, s):
        dux_dx = (u[:, 0, :, 1:] - u[:, 0, :, :-1])[:, :-1, :] / s
        dux_dy = (u[:, 0, 1:, :] - u[:, 0, :-1, :])[:, :, :-1] / s
        duy_dx = (u[:, 1, :, 1:] - u[:, 1, :, :-1])[:, :-1, :] / s
        duy_dy = (u[:, 1, 1:, :] - u[:, 1, :-1, :])[:, :, :-1] / s
        return (1 + dux_dx) * (1 + duy_dy) - dux_dy * duy_dx

    def regulariser(self, u, s):
        gx = (u[..., :, 1:] - u[..., :, :-1]) / s
        gy = (u[..., 1:, :] - u[..., :-1, :]) / s
        diff = (gx ** 2).mean() + (gy ** 2).mean()
        fold = F.relu(0.1 - self.jacobian_det(u, s)).mean()
        return diff, fold

    # ---------------- evaluation helpers
    @torch.no_grad()
    def metrics(self, s=8.0, with_deform=True):
        _, met = self._loss(s, with_deform)
        return met


@torch.no_grad()
def dense_map_px(T: Transform, with_deform=True, chunk_rows=512) -> np.ndarray:
    """H&E pixel coordinate (x,y) for every OCT pixel: (2,H,W) float32."""
    H, W = T.fixed_shape
    X = T.grid_um(T.fixed_px_um)
    out = np.zeros((2, H, W), np.float32)
    for r in range(0, H, chunk_rows):
        P = T.map_px(X[r:r + chunk_rows], with_deform)
        out[:, r:r + chunk_rows] = P.permute(2, 0, 1).cpu().numpy()
    return out


@torch.no_grad()
def warp_image(img: np.ndarray, T: Transform, with_deform=True, mode="bilinear", fill=0.0,
               chunk_rows=1024, device=DEVICE, spacing_um: float | None = None) -> np.ndarray:
    """Warp a native-resolution H&E-space image (H,W) or (H,W,C) into the OCT frame.

    spacing_um=None -> OCT pixel grid; otherwise an exact-spacing grid over the OCT frame.
    """
    squeeze = img.ndim == 2
    a = img[..., None] if squeeze else img
    dt = a.dtype
    src = torch.tensor(a.astype(np.float32), device=device).permute(2, 0, 1)[None] - fill
    sp = T.fixed_px_um if spacing_um is None else spacing_um
    H, W = T.grid_shape_exact(sp)
    out = np.zeros((H, W, a.shape[2]), np.float32)
    for r in range(0, H, chunk_rows):
        X = T.grid_um_exact(sp, (r, r + chunk_rows))
        g = T.px_to_norm(T.map_px(X, with_deform))[None]
        o = F.grid_sample(src, g, mode=mode, padding_mode="zeros", align_corners=False)[0] + fill
        out[r:r + X.shape[0]] = o.permute(1, 2, 0).cpu().numpy()
    del src
    empty_cache()
    if np.issubdtype(dt, np.integer):
        out = np.clip(np.rint(out), np.iinfo(dt).min, np.iinfo(dt).max).astype(dt)
    return out[..., 0] if squeeze else out


@torch.no_grad()
def resample_fixed(img: np.ndarray, fixed_px_um: float, spacing_um: float, shape, mode="bilinear", device=DEVICE) -> np.ndarray:
    """Resample an OCT-grid image onto an exact-spacing grid of the same frame (NaN-aware for floats)."""
    a = img.astype(np.float32)
    nan = np.isnan(a)
    src = torch.tensor(np.nan_to_num(a), device=device)[None, None]
    val = torch.tensor((~nan).astype(np.float32), device=device)[None, None]
    H, W = shape; Hs, Ws = img.shape
    ys = ((torch.arange(H, device=device) + 0.5) * spacing_um) / (Hs * fixed_px_um) * 2 - 1
    xs = ((torch.arange(W, device=device) + 0.5) * spacing_um) / (Ws * fixed_px_um) * 2 - 1
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    g = torch.stack([gx, gy], -1)[None]
    o = F.grid_sample(src * val, g, mode=mode, padding_mode="zeros", align_corners=False)[0, 0]
    v = F.grid_sample(val, g, mode="bilinear", padding_mode="zeros", align_corners=False)[0, 0]
    o = torch.where(v > 0.5, o / v.clamp(min=1e-6), torch.full_like(o, float("nan")))
    return o.cpu().numpy()
