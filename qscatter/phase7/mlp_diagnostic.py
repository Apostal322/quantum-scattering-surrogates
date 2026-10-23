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


def pole_grid(fom, lo_h, hi_h, n: int, E_range):
    """(V, E_r, log Gamma) of the two lowest poles on an n x n height grid."""
    V = np.linspace(lo_h, hi_h, n)
    Er = np.empty((2, n, n))
    lg = np.empty((2, n, n))
    for i, a in enumerate(V):
        for j, b in enumerate(V):
            ps = sorted(find_poles(fom, (a, b), E_range), key=lambda z: z.real)[:2]
            for m, z in enumerate(ps):
                Er[m, i, j], lg[m, i, j] = z.real, np.log(-2 * z.imag)
    return V, Er, lg


def maps_from_grid(V, Er, lg):
    return [(RegularGridInterpolator((V, V), Er[m], method="cubic"), RegularGridInterpolator((V, V), lg[m], method="cubic"))
            for m in range(2)]


def pole_maps(fom, lo_h, hi_h, n: int, E_range):
    """Interpolants of (E_r, log Gamma) of the two lowest poles over the height box."""
    return maps_from_grid(*pole_grid(fom, lo_h, hi_h, n, E_range))


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


_CTX: dict[str, Any] = {}  # shared with forked workers (read-only)


def _fit_task(task):
    """One MLP fit + evaluation; runs in a worker process (one thread)."""
    import torch

    torch.set_num_threads(1)
    c = _CTX
    key, width, depth, seed, history = task
    X, t = c["data"][key]
    ts = time.perf_counter()
    hist: list | None = [] if history else None
    pred, v = mlp.fit_mlp(X, t, c["Xv"], c["tv"], c["lo"], c["hi"], width, c["lr"], c["epochs"], seed=seed,
                          patience=c["patience"], depth=depth, history=hist)
    row = {"data": key, "width": width, "depth": depth, "seed": seed, "val_mse": v, "fit_seconds": time.perf_counter() - ts,
           "test": summarize(pred, c["Xte"], c["tte"], c["near_test"]),
           "train": summarize(pred, X, t, near_mask(X, c["maps"], c["near_factor"]))}
    if hist is not None:
        row["history"] = hist
        row["epochs_run"] = hist[-1][0] if hist else None
    return row


def _checkpointed(path: Path, fn):
    if path.exists():
        return json.loads(path.read_text())
    res = fn()
    path.write_text(json.dumps(res, default=float))
    return res


def run_mlp_diagnostic(cfg: dict, workdir: Path | None = None) -> dict[str, Any]:
    """Stages are checkpointed in `workdir` (default results/phase7/mlp_diagnostic/stages), so an
    interrupted run resumes; independent fits run in parallel (cfg["processes"])."""
    from multiprocessing import get_context

    repo = Path(__file__).resolve().parents[2]
    gen = cfg.get("splits") or json.loads((repo / "configs" / "phase7" / cfg["splits_config"]).read_text())
    work = workdir or repo / "results" / "phase7" / "mlp_diagnostic" / "stages"
    work.mkdir(parents=True, exist_ok=True)
    fom = gaussian_fom(gen["s"], gen["s"], gen["d"], gen["X"], gen["h"])
    sp = make_splits(gen)
    lo, hi = np.array(gen["train_lo"]), np.array(gen["train_hi"])
    E_range = (lo[0], hi[0])
    t0 = time.perf_counter()
    grid = _checkpointed(work / "pole_grid.json",
                         lambda: [a.tolist() for a in pole_grid(fom, lo[1], hi[1], cfg["pole_grid"], E_range)])
    maps = maps_from_grid(*[np.array(a) for a in grid])
    out: dict[str, Any] = {"pole_grid_seconds_this_run": time.perf_counter() - t0}
    near_test = near_mask(sp.test, maps, cfg["near_factor"])
    out["n_test_near"] = int(near_test.sum())
    data = {str(n): sobol(n, lo, hi, gen["seeds"]["design_mlp"]) for n in cfg["budgets"]}
    nmax = str(max(cfg["budgets"]))
    data["resonance_dense_oracle"] = resonance_dense_design(int(nmax), lo, hi, maps, cfg["seed_resonance_design"])
    _CTX.clear()
    _CTX.update(data={k: (X, fom.amplitudes(X)[0]) for k, X in data.items()}, Xv=sp.validation,
                tv=fom.amplitudes(sp.validation)[0], Xte=sp.test, tte=fom.amplitudes(sp.test)[0], near_test=near_test,
                maps=maps, near_factor=cfg["near_factor"], lo=lo, hi=hi, lr=cfg["grid"]["lr"], epochs=cfg["grid"]["epochs"],
                patience=cfg["grid"]["patience"])
    out["train_points_near_resonance"] = {k: int(near_mask(X, maps, cfg["near_factor"]).sum()) for k, X in data.items()}
    pool = get_context("fork").Pool(cfg.get("processes", 4))
    try:
        # 2. bounded search on the largest budget, validation only
        search = _checkpointed(work / "search.json", lambda: pool.map(
            _fit_task, [(nmax, w, dp, 0, True) for w in cfg["grid"]["width"] for dp in cfg["grid"]["depth"]], chunksize=1))
        best = min(search, key=lambda r: r["val_mse"])
        out["search"] = search
        out["selected"] = {"width": best["width"], "depth": best["depth"]}
        # 3-5 and 7: seeds x budgets, and the oracle design, with the selected architecture
        keys = [str(n) for n in cfg["budgets"]] + ["resonance_dense_oracle"]
        rows = _checkpointed(work / "seeded.json", lambda: pool.map(
            _fit_task, [(k, best["width"], best["depth"], s, False) for k in keys for s in cfg["seeds"]], chunksize=1))
    finally:
        pool.close()
        pool.join()
    out["seeded"] = [r for r in rows if r["data"] != "resonance_dense_oracle"]
    out["resonance_dense_oracle_design"] = [r for r in rows if r["data"] == "resonance_dense_oracle"]
    # 6. RBF at the same largest design
    X, t = _CTX["data"][nmax]
    unit = lambda A: (np.atleast_2d(A) - lo) / (hi - lo)  # noqa: E731
    kv = {k: float(np.sqrt(np.mean(np.abs(rbf_predict(rbf_fit(unit(X), t[:, None], k), unit(sp.validation), 1)[:, 0] - _CTX["tv"]) ** 2)))
          for k in cfg["rbf_kernels"]}
    k_best = min(kv, key=kv.get)
    rb = rbf_fit(unit(X), t[:, None], k_best)
    out["rbf_same_points"] = {"kernel_validation_rmse": kv, "kernel": k_best,
                              "test": summarize(lambda A: rbf_predict(rb, unit(A), 1)[:, 0], sp.test, _CTX["tte"], near_test)}
    return out


RUNNERS = {"mlp_diagnostic": run_mlp_diagnostic}
