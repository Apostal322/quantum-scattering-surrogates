"""Audit diagnostic: why does the Phase 7 MLP fail on held-out resonances?

Protocol (frozen in configs/phase7/mlp_diagnostic.json before the run):

1. Same splits, validation and test sets as the generalisation experiment (same seeds).
2. Bounded architecture search on the 4096-point Sobol design: width x depth grid, fixed learning
   rate, longer training than Phase 7 (epochs, patience from the config). Selection by validation
   MSE only. The test set is evaluated once, after selection.
3. The selected configuration is retrained with several seeds (initialisation), at 1024 and 4096
   points; mean, std and worst case are reported.
4. Error decomposition: test points within 10 Gamma of one of the two narrow resonances of their
   (V_L, V_R) versus all others. Pole positions E_r(V_L, V_R), Gamma(V_L, V_R) are computed with the
   FEM on a 7 x 7 height grid and interpolated (log Gamma), which is accurate enough to classify
   points but not used for any fit.
5. Underfitting check: training-set error of the same models.
6. Equal-data baseline: RBF on t at the same 4096 points (kernel chosen on validation).
7. Information diagnostic (NOT a fair baseline: it uses pole positions): an MLP trained on 4096
   points of which half are drawn within +-3 Gamma of a narrow resonance. If it fixes the
   near-resonance error, the original failure is mainly missing information, not expressiveness.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from qscatter.phase7 import mlp
from qscatter.phase7.experiments import point_metrics, rbf_fit, rbf_predict
from qscatter.phase7.fom import gaussian_fom
from qscatter.phase7.protocol import make_splits, sobol
from qscatter.phase7.resonances import find_poles


def pole_maps(fom, lo_h, hi_h, n: int, E_range):
    """Interpolants of (E_r, log Gamma) of the two narrowest-index poles over the height box."""
    V = np.linspace(lo_h, hi_h, n)
    Er = np.empty((2, n, n))
    lg = np.empty((2, n, n))
    for i, a in enumerate(V):
        for j, b in enumerate(V):
            ps = sorted(find_poles(fom, (a, b), E_range), key=lambda z: z.real)[:2]
            for m, z in enumerate(ps):
                Er[m, i, j], lg[m, i, j] = z.real, np.log(-2 * z.imag)
    return [(RegularGridInterpolator((V, V), Er[m], method="cubic"), RegularGridInterpolator((V, V), lg[m], method="cubic")) for m in range(2)]


def near_mask(X, maps, factor):
    m = np.zeros(len(X), bool)
    for fE, fG in maps:
        Er, G = fE(X[:, 1:]), np.exp(fG(X[:, 1:]))
        m |= np.abs(X[:, 0] - Er) < factor * G
    return m


def resonance_dense_design(n, lo, hi, maps, seed):
    rng = np.random.default_rng(seed)
    half = n // 2
    base = sobol(n - half, lo, hi, seed)
    H = rng.uniform(lo[1:], hi[1:], (half, 2))
    which = rng.integers(0, 2, half)
    Er = np.array([maps[w][0](h[None])[0] for w, h in zip(which, H)])
    G = np.array([np.exp(maps[w][1](h[None])[0]) for w, h in zip(which, H)])
    E = Er + G * rng.uniform(-3, 3, half)
    return np.vstack([base, np.column_stack([E, H])])


def summarize(pred, X, t, near):
    th = pred(X)
    if not np.all(np.isfinite(th)):
        raise FloatingPointError("non-finite MLP prediction")
    out = {"all": point_metrics(th, t)}
    if near.any():
        out["near_resonance"] = point_metrics(th[near], t[near])
    out["far_from_resonance"] = point_metrics(th[~near], t[~near])
    out["n_near"] = int(near.sum())
    return out


def run_mlp_diagnostic(cfg: dict) -> dict[str, Any]:
    gen = json.loads((Path(__file__).resolve().parents[2] / "configs" / "phase7" / cfg["splits_config"]).read_text())
    fom = gaussian_fom(gen["s"], gen["s"], gen["d"], gen["X"], gen["h"])
    sp = make_splits(gen)
    lo, hi = np.array(gen["train_lo"]), np.array(gen["train_hi"])
    E_range = (lo[0], hi[0])
    t_val = fom.amplitudes(sp.validation)[0]
    t_test = fom.amplitudes(sp.test)[0]
    t0 = time.perf_counter()
    maps = pole_maps(fom, lo[1], hi[1], cfg["pole_grid"], E_range)
    pole_seconds = time.perf_counter() - t0
    near_test = near_mask(sp.test, maps, cfg["near_factor"])
    out: dict[str, Any] = {"pole_map_seconds": pole_seconds, "n_test_near": int(near_test.sum())}
    designs = {n: sobol(n, lo, hi, gen["seeds"]["design_mlp"]) for n in cfg["budgets"]}
    data = {n: (X, fom.amplitudes(X)[0]) for n, X in designs.items()}
    out["train_points_near_resonance"] = {str(n): int(near_mask(X, maps, cfg["near_factor"]).sum()) for n, (X, _) in data.items()}
    # 2. bounded search on the largest budget, validation only
    nmax = max(cfg["budgets"])
    X, t = data[nmax]
    search = []
    for width in cfg["grid"]["width"]:
        for depth in cfg["grid"]["depth"]:
            ts = time.perf_counter()
            hist: list = []
            pred, v = mlp.fit_mlp(X, t, sp.validation, t_val, lo, hi, width, cfg["grid"]["lr"], cfg["grid"]["epochs"],
                                  seed=0, patience=cfg["grid"]["patience"], depth=depth, history=hist)
            search.append({"width": width, "depth": depth, "val_mse": v, "seconds": time.perf_counter() - ts,
                           "epochs_run": hist[-1][0] if hist else None, "history": hist})
    best = min(search, key=lambda r: r["val_mse"])
    out["search"] = search
    out["selected"] = {"width": best["width"], "depth": best["depth"]}
    # 3-5. seeds, budgets, decomposition, train error
    rows = []
    for n, (Xn, tn) in data.items():
        for seed in cfg["seeds"]:
            ts = time.perf_counter()
            pred, v = mlp.fit_mlp(Xn, tn, sp.validation, t_val, lo, hi, best["width"], cfg["grid"]["lr"], cfg["grid"]["epochs"],
                                  seed=seed, patience=cfg["grid"]["patience"], depth=best["depth"])
            near_tr = near_mask(Xn, maps, cfg["near_factor"])
            rows.append({"n": n, "seed": seed, "val_mse": v, "fit_seconds": time.perf_counter() - ts,
                         "test": summarize(pred, sp.test, t_test, near_test),
                         "train": summarize(pred, Xn, tn, near_tr)})
    out["seeded"] = rows
    # 6. RBF at the same 4096 points
    unit = lambda A: (np.atleast_2d(A) - lo) / (hi - lo)  # noqa: E731
    kv = {}
    for k in cfg["rbf_kernels"]:
        kv[k] = float(np.sqrt(np.mean(np.abs(rbf_predict(rbf_fit(unit(X), t[:, None], k), unit(sp.validation), 1)[:, 0] - t_val) ** 2)))
    k_best = min(kv, key=kv.get)
    rb = rbf_fit(unit(X), t[:, None], k_best)
    out["rbf_same_points"] = {"kernel_validation_rmse": kv, "kernel": k_best,
                              "test": summarize(lambda A: rbf_predict(rb, unit(A), 1)[:, 0], sp.test, t_test, near_test)}
    # 7. information diagnostic
    Xr = resonance_dense_design(nmax, lo, hi, maps, cfg["seed_resonance_design"])
    tr = fom.amplitudes(Xr)[0]
    info = []
    for seed in cfg["seeds"]:
        pred, v = mlp.fit_mlp(Xr, tr, sp.validation, t_val, lo, hi, best["width"], cfg["grid"]["lr"], cfg["grid"]["epochs"],
                              seed=seed, patience=cfg["grid"]["patience"], depth=best["depth"])
        info.append({"seed": seed, "val_mse": v, "test": summarize(pred, sp.test, t_test, near_test)})
    out["resonance_dense_oracle_design"] = info
    return out


RUNNERS = {"mlp_diagnostic": run_mlp_diagnostic}
