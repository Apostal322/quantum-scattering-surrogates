"""Experiment F: robustness of the reduced basis and the energy-only baselines.

All studies run along fixed-height energy lines (the heights of the first test scans of the
generalisation splits), with errors measured against the FEM at the production mesh:

F1 noise:      complex Gaussian noise on the t data (AAA, PCHIP) or on the snapshot vectors (RB),
               relative size sigma, several seeds; counts AAA poles that match no FEM pole.
F2 solver:     training data from coarser meshes (t values for AAA; snapshots interpolated to the
               fine mesh for RB, which is then projected on the fine operator).
F3 jitter:     uniform energy designs perturbed by +-delta of the spacing (AAA, PCHIP, RB on the
               same jittered design).
F4 doublets:   a symmetric triple-Gaussian barrier (two coupled wells) whose middle height sets the
               splitting of near-degenerate resonance pairs.
F5 precision:  the reduced system solved in complex64 instead of complex128.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from qscatter.phase6.experiments import Scan, scan_metrics
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy
from qscatter.phase7.fom import AffineFOM, gaussian, gaussian_fom, truncation_halfwidth
from qscatter.phase7.protocol import make_splits
from qscatter.phase7.resonances import find_poles
from qscatter.phase7.slice1d import _aaa, aaa_adaptive_design, pole_errors


def line_scan(fom: AffineFOM, heights, E_range, n_uniform: int, n_window: int) -> Scan:
    poles = [z for z in find_poles(fom, tuple(heights), tuple(E_range)) if E_range[0] <= z.real <= E_range[1]]
    grids = [np.linspace(*E_range, n_uniform)]
    for z in poles:
        G = -2 * z.imag
        grids.append(np.linspace(max(E_range[0], z.real - 20 * G), min(E_range[1], z.real + 20 * G), n_window))
    E = np.unique(np.concatenate(grids))
    t = fom.amplitudes(np.column_stack([E] + [np.full_like(E, v) for v in heights]))[0]
    return Scan(dict(zip(fom.params, heights)), E, t, poles)


def summarize(sc: Scan, f: Callable[[np.ndarray], np.ndarray], T_fom, poles_hat=None) -> dict[str, Any]:
    th = f(sc.E)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = scan_metrics(sc, th, lambda E: np.abs(f(E)) ** 2, T_fom)
    res = m.pop("resonances")
    out = {"max_abs_dt": m["max_abs_dt"], "max_abs_dT": m["max_abs_dT"], "rmse_T": m["rmse_T"],
           "spurious_peaks": m["spurious_peaks"],
           "recall": float(np.mean([r["resolved"] for r in res])) if res else None,
           "max_peak_shift_over_gamma": max((r["peak_shift_over_gamma"] for r in res), default=None)}
    if poles_hat is not None:
        lo, hi = sc.E[0], sc.E[-1]
        near = [p for p in poles_hat if lo < p.real < hi and -1.0 < p.imag < 0]
        out["poles_matched"] = float(np.mean([e["matched"] for e in pole_errors(near, sc.poles)])) if sc.poles else None
        out["unmatched_poles_near_axis"] = int(sum(all(abs(p - z) > max(3 * (-2 * z.imag), 0.05) for z in sc.poles) for p in near))
    return out


def _aaa_fn(E, t):
    r = _aaa(E, t)
    return (lambda Q: r(np.sqrt(np.asarray(Q, complex)))), list(np.asarray(r.poles()) ** 2)


def run_robustness(cfg: dict) -> dict[str, Any]:
    gen = cfg.get("splits") or json.loads((Path(__file__).resolve().parents[2] / "configs" / "phase7" / cfg["splits_config"]).read_text())
    s, d, X, h = gen["s"], gen["d"], gen["X"], gen["h"]
    fom = gaussian_fom(s, s, d, X, h)
    sp = make_splits(gen)
    E_range = (gen["train_lo"][0], gen["train_hi"][0])
    E_cand = np.linspace(*E_range, cfg["n_energy_candidates"])
    n = cfg["budget"]
    seeds = cfg["seeds"]
    out: dict[str, Any] = {"splits_config": gen, "lines": []}
    for heights in sp.scan_heights_test[: cfg["n_lines"]]:
        sc = line_scan(fom, heights, E_range, cfg["scan"]["n_uniform"], cfg["scan"]["n_window"])

        def mu_of(E, hh=heights):
            E = np.atleast_1d(E)
            return np.column_stack([E, np.full(E.shape, hh[0]), np.full(E.shape, hh[1])])

        def t_fom(E):
            return fom.amplitudes(mu_of(E))[0]

        def T_fom(E):
            return np.abs(t_fom(E)) ** 2

        def rb_fn(snaps):
            rm = ReducedModel(fom, orthonormalize(snaps)[0])
            return lambda E: rm.amplitudes(mu_of(E), residual=False)[0]

        greedy = residual_greedy(fom, mu_of(E_cand), n, mu_of(np.mean(E_range))[0])
        E_u = np.linspace(*E_range, n)
        t_u = t_fom(E_u)
        rec: dict[str, Any] = {"heights": heights, "poles": [[z.real, z.imag] for z in sc.poles]}

        # F1 noise
        f1 = []
        for sig in cfg["noise_levels"]:
            for seed in seeds:
                rng = np.random.default_rng(seed)
                cn = lambda shape: (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)  # noqa: E731
                noisy = lambda E: t_fom(E) + sig * cn(np.shape(E))  # noqa: E731
                t_noisy = t_u + sig * cn(n)
                fa, pa = _aaa_fn(E_u, t_noisy)
                Ea, ta = aaa_adaptive_design(noisy, E_cand, n)
                fd, pd = _aaa_fn(Ea, ta)
                S = greedy.snapshots
                S_noisy = S + sig * np.linalg.norm(S, axis=0) / np.sqrt(S.shape[0]) * cn(S.shape)
                p = (PchipInterpolator(E_u, t_noisy.real), PchipInterpolator(E_u, t_noisy.imag))
                f1.append({"sigma": sig, "seed": seed,
                           "aaa_uniform": summarize(sc, fa, T_fom, pa),
                           "aaa_adaptive": summarize(sc, fd, T_fom, pd),
                           "pchip_uniform": summarize(sc, lambda E, p=p: p[0](E) + 1j * p[1](E), T_fom),
                           "rb_greedy_noisy_snapshots": summarize(sc, rb_fn(S_noisy), T_fom)})
        rec["noise"] = f1

        # F2 solver inaccuracy (coarse-mesh training data)
        f2 = []
        for hc in cfg["coarse_meshes"]:
            fc = gaussian_fom(s, s, d, X, hc)
            Sc = fc.solve(greedy.selected).T  # same parameters as the fine greedy
            S_interp = np.column_stack([np.interp(fom.nodes, fc.nodes, c.real) + 1j * np.interp(fom.nodes, fc.nodes, c.imag) for c in Sc.T])
            tc = fc.amplitudes(mu_of(E_u))[0]
            fa, pa = _aaa_fn(E_u, tc)
            coarse_fom = lambda E, fc=fc: fc.amplitudes(mu_of(E))[0]  # noqa: E731
            f2.append({"h_coarse": hc, "fom_coarse_vs_fine": summarize(sc, coarse_fom, T_fom),
                       "aaa_uniform_coarse_data": summarize(sc, fa, T_fom, pa),
                       "rb_coarse_snapshots_fine_operator": summarize(sc, rb_fn(S_interp), T_fom)})
        rec["solver_inaccuracy"] = f2

        # F3 energy-grid jitter
        f3 = []
        dE = E_u[1] - E_u[0]
        for delta in cfg["jitter"]:
            for seed in seeds:
                Ej = np.sort(np.clip(E_u + delta * dE * np.random.default_rng(seed).uniform(-1, 1, n), *E_range))
                tj = t_fom(Ej)
                fa, pa = _aaa_fn(Ej, tj)
                p = (PchipInterpolator(Ej, tj.real), PchipInterpolator(Ej, tj.imag))
                f3.append({"delta": delta, "seed": seed, "aaa": summarize(sc, fa, T_fom, pa),
                           "pchip": summarize(sc, lambda E, p=p: p[0](E) + 1j * p[1](E), T_fom),
                           "rb_same_design": summarize(sc, rb_fn(fom.solve(mu_of(Ej)).T), T_fom)})
        rec["jitter"] = f3

        # F5 precision of the online solve
        rm = ReducedModel(fom, orthonormalize(greedy.snapshots)[0])
        mu_s = mu_of(sc.E)
        th, tb = fom.theta(mu_s)
        Ar = np.einsum("nq,qij->nij", th, rm.Ar)
        rhs = tb * rm.fr[None, :]
        prec = {}
        for dt in ("complex128", "complex64"):
            c = np.linalg.solve(Ar.astype(dt), rhs.astype(dt)[..., None])[..., 0].astype(complex)
            t_hat, _ = fom.observables(mu_s, c @ rm.end0, c @ rm.endL)
            prec[dt] = float(np.max(np.abs(t_hat - sc.t_fom)))
        prec["cond_reduced_max"] = float(np.max(np.linalg.cond(Ar)))
        rec["precision"] = prec
        out["lines"].append(rec)

    # F4 near-degenerate resonances: symmetric triple Gaussian barrier
    tg = cfg["triple"]
    sd, st = tg["spacing"], tg["s"]
    Xt = truncation_halfwidth(2 * sd, st, tg["rel_cut"])
    ftri = AffineFOM({"V1": gaussian(-sd, st), "V2": gaussian(0.0, st), "V3": gaussian(sd, st)}, X=Xt, h_target=h)
    E_rt = tuple(tg["E_range"])
    E_ct = np.linspace(*E_rt, cfg["n_energy_candidates"])
    f4 = []
    for Vm in tg["middle_heights"]:
        hh = (tg["outer"], Vm, tg["outer"])
        sc = line_scan(ftri, hh, E_rt, cfg["scan"]["n_uniform"], cfg["scan"]["n_window"])
        mu_t = lambda E, hh=hh: np.column_stack([np.atleast_1d(E)] + [np.full(np.shape(np.atleast_1d(E)), v) for v in hh])  # noqa: E731
        t_tri = lambda E, mu_t=mu_t: ftri.amplitudes(mu_t(E))[0]  # noqa: E731
        T_tri = lambda E, t_tri=t_tri: np.abs(t_tri(E)) ** 2  # noqa: E731
        zs = sorted(sc.poles, key=lambda z: z.real)
        pairs = [{"E": [a.real, b.real], "gamma": [-2 * a.imag, -2 * b.imag],
                  "splitting_over_mean_gamma": (b.real - a.real) / (-(a.imag + b.imag))} for a, b in zip(zs[:-1], zs[1:])]
        g = residual_greedy(ftri, mu_t(E_ct), max(tg["budgets"]), mu_t(np.mean(E_rt))[0])
        rows = []
        for nb in tg["budgets"]:
            rm = ReducedModel(ftri, orthonormalize(g.snapshots[:, :nb])[0])
            Ea, ta = aaa_adaptive_design(t_tri, E_ct, nb)
            fd, pd = _aaa_fn(Ea, ta)
            Eu = np.linspace(*E_rt, nb)
            fa, pa = _aaa_fn(Eu, t_tri(Eu))
            rows.append({"n": nb,
                         "rb_greedy": summarize(sc, lambda E, rm=rm, mu_t=mu_t: rm.amplitudes(mu_t(E), residual=False)[0], T_tri),
                         "aaa_adaptive": summarize(sc, fd, T_tri, pd),
                         "aaa_uniform": summarize(sc, fa, T_tri, pa)})
        f4.append({"heights": list(hh), "poles": [[z.real, z.imag] for z in zs], "adjacent_pairs": pairs, "rows": rows})
    out["doublets"] = {"geometry": tg, "X": Xt, "configs": f4}
    return out


RUNNERS = {"robustness": run_robustness}
