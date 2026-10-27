"""Experiment C, 1-D part: energy-only methods on fixed-height test scans.

AAA and PCHIP approximate t(E) along one energy line and cannot generalise across barrier heights,
so they are compared with a reduced basis built for the same line (heights fixed), at equal numbers
of FOM queries. Information each method may use:

- PCHIP / AAA (uniform):   n FOM values t(E_i) on a uniform energy grid (the declared design).
- AAA (adaptive):          n FOM values t(E_i); the next energy maximises the disagreement of the
                           last two AAA approximants on a candidate grid (t queries only).
- RB greedy (Galerkin):    n FOM wave functions psi(E_i) chosen by the residual indicator, which
                           needs the affine FOM operator (not only t).

The test scans and their heights are the ones of the generalisation experiment (same seeds), and no
choice is made on them. Poles of every approximant are compared with the FEM poles, which isolates
E_r / Gamma accuracy from peak-finding on a grid.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import AAA, PchipInterpolator

from qscatter.phase6.experiments import scan_metrics
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy
from qscatter.phase7.experiments import make_scan
from qscatter.phase7.fom import gaussian_fom
from qscatter.phase7.protocol import make_splits


def secant_pole(inv_t: Callable[[complex], complex], guess: complex, step_tol: float = 1e-11, maxiter: int = 60,
                depth: float = 1e-5) -> complex | None:
    """Root of a (surrogate's) 1/t(E) near `guess`, or None if no root was reached.

    The best iterate is accepted only if Im z < 0 and |1/t(z)| < depth * |1/t(Re z)| (the same
    scale-free pole test as `AffineFOM.find_pole`), so a stalled iteration is never reported as a pole.
    """
    z0 = complex(guess)
    z1 = z0 + 1e-6 * max(1.0, abs(z0))
    f0, f1 = inv_t(z0), inv_t(z1)
    best = (abs(f1), z1) if abs(f1) < abs(f0) else (abs(f0), z0)
    for _ in range(maxiter):
        if f1 == f0 or not np.isfinite(f1):
            break
        z0, f0, z1 = z1, f1, z1 - f1 * (z1 - z0) / (f1 - f0)
        f1 = inv_t(z1)
        if abs(f1) < best[0]:
            best = (abs(f1), z1)
        if abs(z1 - z0) < step_tol * max(1.0, abs(z1)):
            break
    f_best, z = best
    if not (z.imag < 0 and z.real > 0):
        return None
    return z if f_best < depth * abs(inv_t(complex(z.real, 0.0))) else None


def _aaa(E: np.ndarray, t: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AAA(np.sqrt(E), t, rtol=1e-13, max_terms=len(E))


def aaa_adaptive_design(t_of: Callable[[np.ndarray], np.ndarray], E_cand: np.ndarray, n: int, n_start: int = 4):
    """Greedy energy design from t queries only (disagreement of consecutive AAA fits)."""
    E = list(np.linspace(E_cand[0], E_cand[-1], n_start))
    t = list(t_of(np.array(E)))
    prev = None
    while len(E) < n:
        r = _aaa(np.array(E), np.array(t))
        cur = r(np.sqrt(E_cand))
        score = np.abs(cur - prev) if prev is not None else np.abs(cur)
        score[~np.isfinite(score)] = np.inf  # a pole of either fit on the real axis: sample there
        score[np.isin(E_cand, E)] = -np.inf
        j = int(np.argmax(score))
        E.append(float(E_cand[j]))
        t.append(t_of(np.array([E_cand[j]]))[0])
        prev = cur
    return np.array(E), np.array(t)


def pole_errors(poles_hat: list[complex], poles_ref: list[complex]) -> list[dict[str, float]]:
    out = []
    for z in poles_ref:
        G = -2 * z.imag
        if not poles_hat:
            out.append({"E_r": z.real, "gamma": G, "matched": False})
            continue
        p = min(poles_hat, key=lambda q: abs(q - z))
        out.append({"E_r": z.real, "gamma": G, "matched": bool(abs(p - z) < 2 * G),
                    "err_Er_over_gamma": abs(p.real - z.real) / G, "rel_err_gamma": abs(-2 * p.imag - G) / G})
    return out


def run_slice1d(cfg: dict) -> dict[str, Any]:
    gen = cfg.get("splits") or json.loads((Path(__file__).resolve().parents[2] / "configs" / "phase7" / cfg["splits_config"]).read_text())
    s, d, X, h = gen["s"], gen["d"], gen["X"], gen["h"]
    fom = gaussian_fom(s, s, d, X, h)
    sp = make_splits(gen)
    E_range = (gen["train_lo"][0], gen["train_hi"][0])
    E_cand = np.linspace(*E_range, cfg["n_energy_candidates"])
    out: list[dict[str, Any]] = []
    for heights in sp.scan_heights_test:
        sc = make_scan(fom, heights, E_range, cfg["scan"]["n_uniform"], cfg["scan"]["n_window"])

        def mu_of(E, hh=heights):
            E = np.atleast_1d(E)
            return np.column_stack([E, np.full(E.shape, hh[0]), np.full(E.shape, hh[1])])

        def t_fom(E):
            return fom.amplitudes(mu_of(E))[0]

        def T_fom(E):
            return np.abs(t_fom(E)) ** 2

        greedy = residual_greedy(fom, mu_of(E_cand), max(cfg["budgets"]), mu_of(np.mean(E_range))[0])
        rec: dict[str, Any] = {"heights": heights, "poles": [[z.real, z.imag] for z in sc.poles], "rows": []}
        for n in cfg["budgets"]:
            E_u = np.linspace(*E_range, n)
            t_u = t_fom(E_u)
            t0 = time.perf_counter()
            E_a, t_a = aaa_adaptive_design(t_fom, E_cand, n)
            design_seconds = time.perf_counter() - t0
            V, _ = orthonormalize(greedy.snapshots[:, :n])
            rm = ReducedModel(fom, V)
            r_u, r_a = _aaa(E_u, t_u), _aaa(E_a, t_a)
            pch = (PchipInterpolator(E_u, t_u.real), PchipInterpolator(E_u, t_u.imag))
            methods: dict[str, tuple[Callable, list[complex] | None]] = {
                "pchip_uniform": (lambda E, p=pch: p[0](E) + 1j * p[1](E), None),
                "aaa_uniform": (lambda E, r=r_u: r(np.sqrt(np.asarray(E, complex))), list(np.asarray(r_u.poles()) ** 2)),
                "aaa_adaptive": (lambda E, r=r_a: r(np.sqrt(np.asarray(E, complex))), list(np.asarray(r_a.poles()) ** 2)),
                "rb_greedy": (lambda E, rm=rm: rm.amplitudes(mu_of(E), residual=False)[0], None),
            }
            row: dict[str, Any] = {"n": n, "aaa_adaptive_design_seconds": design_seconds}
            for name, (f, raw_poles) in methods.items():
                th = f(sc.E)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m = scan_metrics(sc, th, lambda E, f=f: np.abs(f(E)) ** 2, T_fom)
                m["mae_T"] = float(np.trapezoid(np.abs(np.abs(th) ** 2 - np.abs(sc.t_fom) ** 2), sc.E) / (sc.E[-1] - sc.E[0]))
                # poles: AAA has explicit poles; RB poles are roots of 1/t_RB(E) continued to complex E
                if name == "rb_greedy":
                    def inv(z, rm=rm, hh=heights):
                        return 1.0 / rm.amplitudes(np.array([[z, hh[0], hh[1]]], complex), residual=False)[0][0]
                    ph = [p for p in (secant_pole(inv, complex(z.real, 0.5 * z.imag)) for z in sc.poles) if p is not None]
                elif raw_poles is not None:
                    ph = [p for p in raw_poles if p.imag < 0 and E_range[0] - 1 < p.real < E_range[1] + 1]
                else:
                    ph = None
                m["pole_errors"] = pole_errors(ph, sc.poles) if ph is not None else None
                if raw_poles is not None:  # poles of the rational fit that match no FEM pole (within 3 Gamma or 0.05)
                    near_axis = [p for p in raw_poles if E_range[0] < p.real < E_range[1] and -1.0 < p.imag < 0]
                    m["unmatched_poles_near_axis"] = int(sum(
                        all(abs(p - z) > max(3 * (-2 * z.imag), 0.05) for z in sc.poles) for p in near_axis))
                row[name] = m
            rec["rows"].append(row)
        out.append(rec)
    return {"splits_config": gen, "scans": out}


RUNNERS = {"slice1d": run_slice1d}
