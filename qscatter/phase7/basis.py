"""Experiment D: how many basis functions does the reduced model need, and what drives it?

Candidate drivers are confounded in a plain parameter sweep (higher barriers narrow the resonances
and shift them; longer wells add resonances and change their widths). Each sweep below moves one
geometric knob at a time along a fixed-height energy line, so the drivers are varied as separately
as the physics allows:

- `V0`    barrier height at fixed geometry: roughly the same number of resonances, widths shrink
          exponentially;
- `d`     separation at fixed height and width: more resonances (longer well);
- `s`     barrier width at fixed height: narrower resonances (thicker barriers) at a similar count;
- `E_max` energy window at fixed geometry: more resonances, the same physics per resonance.

For every configuration: the FEM poles in the window, the numerical rank of a dense snapshot set
(singular values), and the greedy basis size needed for given validation accuracies of t. A
log-linear fit of n against (number of resonances, log(1/Gamma_min)) over all configurations then
shows which factor explains the size; that fit is descriptive, not a law.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy
from qscatter.phase7.fom import gaussian_fom, truncation_halfwidth
from qscatter.phase7.resonances import find_poles


def _line(E: np.ndarray, V0: float) -> np.ndarray:
    return np.column_stack([E, np.full_like(E, V0), np.full_like(E, V0)])


def study(s: float, d: float, V0: float, E_max: float, cfg: dict) -> dict[str, Any]:
    E_min = cfg["E_min"]
    X = truncation_halfwidth(d, s, cfg["rel_cut"])
    fom = gaussian_fom(s, s, d, X, cfg["h"])
    poles = [z for z in find_poles(fom, (V0, V0), (E_min, E_max)) if E_min <= z.real <= E_max]
    gam = [-2 * z.imag for z in poles]
    E_c = np.linspace(E_min, E_max, cfg["n_candidates"])
    rng = np.random.default_rng(cfg["seed_validation"])
    E_v = np.sort(rng.uniform(E_min, E_max, cfg["n_validation"]))
    # validation points also inside every resonance (+-Gamma), where errors are largest
    E_v = np.concatenate([E_v] + [z.real + g * np.array([-1, -0.5, 0, 0.5, 1]) for z, g in zip(poles, gam)])
    E_v = E_v[(E_v > E_min) & (E_v < E_max)]
    t_v = fom.amplitudes(_line(E_v, V0))[0]
    # numerical rank of a dense snapshot set
    S_dense = fom.solve(_line(np.linspace(E_min, E_max, cfg["n_dense"]), V0)).T
    sv = np.linalg.svd(S_dense, compute_uv=False)
    rank = {str(tol): int(np.sum(sv > tol * sv[0])) for tol in cfg["rank_tols"]}
    t0 = time.perf_counter()
    g = residual_greedy(fom, _line(E_c, V0), cfg["n_max"], _line(np.array([0.5 * (E_min + E_max)]), V0)[0])
    greedy_seconds = time.perf_counter() - t0
    errs = []
    for n in range(1, cfg["n_max"] + 1):
        rm = ReducedModel(fom, orthonormalize(g.snapshots[:, :n])[0])
        t_hat = rm.amplitudes(_line(E_v, V0), residual=False)[0]
        errs.append(float(np.max(np.abs(t_hat - t_v))))
    errs_arr = np.array(errs)
    n_for = {str(tol): (int(np.argmax(errs_arr < tol)) + 1 if np.any(errs_arr < tol) else None) for tol in cfg["accuracy_tols"]}
    return {
        "s": s, "d": d, "V0": V0, "E_max": E_max, "X": X, "nodes": int(fom.nodes.size),
        "poles": [[z.real, z.imag] for z in poles], "n_resonances": len(poles),
        "gamma_min": min(gam) if gam else None, "gamma_max": max(gam) if gam else None,
        "snapshot_rank": rank, "singular_values": (sv[: cfg["n_max"] + 10] / sv[0]).tolist(),
        "greedy_max_err_t": errs, "greedy_indicator": g.max_indicator, "n_for_accuracy": n_for,
        "greedy_seconds": greedy_seconds,
    }


def run_basis(cfg: dict) -> dict[str, Any]:
    b = cfg["base"]
    sweeps: dict[str, list[dict[str, Any]]] = {}
    for knob, values in cfg["sweeps"].items():
        sweeps[knob] = []
        for v in values:
            p = dict(b, **{knob: v})
            sweeps[knob].append(study(p["s"], p["d"], p["V0"], p["E_max"], cfg))
    # descriptive fit over all configurations: n(tol) ~ a + b * N_res + c * log10(1/Gamma_min)
    fits = {}
    allr = [r for rs in sweeps.values() for r in rs if r["gamma_min"]]
    for tol in map(str, cfg["accuracy_tols"]):
        rows = [r for r in allr if r["n_for_accuracy"][tol] is not None]
        if len(rows) < 4:
            continue
        A = np.column_stack([np.ones(len(rows)), [r["n_resonances"] for r in rows], [np.log10(1 / r["gamma_min"]) for r in rows]])
        y = np.array([r["n_for_accuracy"][tol] for r in rows], float)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ coef
        fits[tol] = {"coef_const_Nres_log10invGamma": coef.tolist(), "n_configs": len(rows),
                     "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
                     "corr_n_vs_Nres": float(np.corrcoef(A[:, 1], y)[0, 1]),
                     "corr_n_vs_log_invGamma": float(np.corrcoef(A[:, 2], y)[0, 1]),
                     "corr_Nres_vs_log_invGamma": float(np.corrcoef(A[:, 1], A[:, 2])[0, 1])}
    return {"sweeps": sweeps, "fits": fits}


RUNNERS = {"basis": run_basis}
