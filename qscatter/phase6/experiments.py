"""Phase 6 experiments: projection-based reduced basis vs. data-driven surrogates.

Three experiments, each configured by a JSON file in configs/phase6/ and written by
scripts/phase6/run.py to results/phase6/<name>/results.json:

- rb_1d:   the Stage B double barrier, parameter E only (comparable with Phases 1-5).
- rb_3d:   the same geometry with parameters (E, V_L, V_R): asymmetric, three-dimensional.
- nwidth:  what sets the reduced-basis size: singular-value decay of large snapshot sets while
           the resonance widths (barrier height) or the number of resonances (well width) vary.

Data hygiene: candidate sets (used by the greedy indicator), training samples (FOM solves) and
test sets are generated from different seeds/grids; test FOM solutions are never used for
selection or fitting. Accuracy is measured against the FOM (the FOM's own error against the exact
S-matrix solver is reported separately).
"""
from __future__ import annotations

import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import scipy
from scipy.optimize import brentq, minimize_scalar
from scipy.signal import find_peaks
from scipy.stats import qmc, spearmanr

from qscatter.phase6 import baselines as bl
from qscatter.phase6.fom import FOM, double_barrier_layers
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy
from qscatter.physics.poles import find_poles
from qscatter.physics.smatrix import compute_scattering

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ provenance


def provenance(config: dict) -> dict:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                                  check=False).stdout.strip()
        except OSError:
            return None

    dirty = git("status", "--porcelain", "--", "qscatter", "configs", "scripts")
    return {
        "config": config,
        "git_commit": git("rev-parse", "HEAD"),
        "code_modified_since_commit": bool(dirty),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "machine": platform.machine(),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ------------------------------------------------------------------ test scans


@dataclass
class Scan:
    """A resonance-resolved energy scan at fixed barrier heights (test data only)."""

    heights: dict[str, float]
    E: np.ndarray
    t_fom: np.ndarray
    poles: list[complex]

    def mu(self, names: list[str]) -> np.ndarray:
        cols = [self.E] + [np.full_like(self.E, self.heights[p]) for p in names[1:]]
        return np.column_stack(cols)


def make_scan(fom: FOM, heights: dict[str, float], E_range: tuple[float, float], n_uniform: int, n_window: int) -> Scan:
    pot = fom.layers.potential(heights)
    lo, hi = E_range
    poles = [z for z in find_poles(pot, E_range) if lo <= z.real <= hi]
    grids = [np.linspace(lo, hi, n_uniform)]
    for z in poles:
        G = -2 * z.imag
        grids.append(np.linspace(max(lo, z.real - 20 * G), min(hi, z.real + 20 * G), n_window))
    E = np.unique(np.concatenate(grids))
    names = fom.param_names
    mu = np.column_stack([E] + [np.full_like(E, heights[p]) for p in names[1:]])
    t, _ = fom.amplitudes(mu)
    return Scan(heights, E, t, poles)


def l2_rmse(E: np.ndarray, err: np.ndarray) -> float:
    return float(np.sqrt(np.trapezoid(err**2, E) / (E[-1] - E[0])))


def peak_position(T_func: Callable[[np.ndarray], np.ndarray], z: complex) -> tuple[float, float]:
    """Refined maximum of T within +-3 Gamma of a pole's real part: (E_peak, T_peak)."""
    G = -2 * z.imag
    opt = minimize_scalar(
        lambda e: -float(T_func(np.array([e]))[0]),
        bounds=(z.real - 3 * G, z.real + 3 * G),
        method="bounded",
        options={"xatol": 1e-12 * max(1.0, z.real)},
    )
    return float(opt.x), float(-opt.fun)


def half_max_width(
    T_func: Callable[[np.ndarray], np.ndarray],
    e_peak: float,
    T_peak: float,
    G: float,
    E_range: tuple[float, float],
) -> float | None:
    """Full width at half maximum of the peak at e_peak (Brent roots of T - T_peak/2).

    The search stays within 10 Gamma of the peak and inside E_range (energies outside the scan,
    in particular E <= 0, are never evaluated). Returns None when T does not fall below T_peak/2
    there on both sides (e.g. a broad resonance on a high background): FWHM undefined.
    """
    def f(e: float) -> float:
        return float(T_func(np.array([e]))[0]) - 0.5 * T_peak

    edges = []
    for side in (-1, 1):
        far = float(np.clip(e_peak + side * 10 * G, *E_range))
        if far == e_peak:
            return None
        if f(far) >= 0:
            return None
        lo, hi = sorted((e_peak, far))
        edges.append(brentq(f, lo, hi, xtol=1e-12 * max(1.0, e_peak)))
    return float(edges[1] - edges[0])


def spurious_peaks(E: np.ndarray, T: np.ndarray, poles: list[complex]) -> int:
    """Peaks of T with prominence > 0.05 farther than 3 Gamma from every exact pole."""
    idx, _ = find_peaks(T, prominence=0.05)
    count = 0
    for i in idx:
        if all(abs(E[i] - z.real) > 3 * (-2 * z.imag) for z in poles):
            count += 1
    return count


def scan_metrics(scan: Scan, t_hat: np.ndarray, T_func_hat, T_func_fom) -> dict[str, Any]:
    T, Th = np.abs(scan.t_fom) ** 2, np.abs(t_hat) ** 2
    out: dict[str, Any] = {
        "rmse_T": l2_rmse(scan.E, Th - T),
        "max_abs_dT": float(np.max(np.abs(Th - T))),
        "max_abs_dt": float(np.max(np.abs(t_hat - scan.t_fom))),
        "spurious_peaks": spurious_peaks(scan.E, Th, scan.poles),
        "spurious_peaks_fom": spurious_peaks(scan.E, T, scan.poles),
        "resonances": [],
    }
    for z in scan.poles:
        G = -2 * z.imag
        e_ref, T_ref = peak_position(T_func_fom, z)
        if T_ref < 0.05:  # not a visible peak in this configuration (e.g. strongly asymmetric)
            continue
        e_hat, T_pk = peak_position(T_func_hat, z)
        w = (scan.E > z.real - 2 * G) & (scan.E < z.real + 2 * G)
        found = bool(w.any() and np.max(np.abs(Th[w] - T[w])) <= 0.1 * T_ref)
        rng = (float(scan.E[0]), float(scan.E[-1]))
        W_ref = half_max_width(T_func_fom, e_ref, T_ref, G, rng)
        W_hat = half_max_width(T_func_hat, e_hat, T_pk, G, rng) if W_ref is not None else None
        out["resonances"].append(
            {
                "E_pole": z.real,
                "gamma": G,
                "T_peak_fom": T_ref,
                "peak_shift_over_gamma": abs(e_hat - e_ref) / G,
                "peak_height_error": abs(T_pk - T_ref),
                "fwhm_fom": W_ref,
                "fwhm_rel_error": (abs(W_hat - W_ref) / W_ref) if (W_ref and W_hat) else None,
                "fwhm_missing": W_ref is not None and W_hat is None,
                "resolved": found,
            }
        )
    return out


# ----------------------------------------------------------------- the methods


def evaluate_methods(
    fom: FOM,
    mu_train: np.ndarray,
    snaps: np.ndarray,
    t_train: np.ndarray,
    test_mu: np.ndarray,
    t_test: np.ndarray,
    scans: list[Scan],
    lo: np.ndarray,
    hi: np.ndarray,
    one_d: bool,
) -> dict[str, dict[str, Any]]:
    """All methods on one training sample; returns per-method metrics."""
    V, sv = orthonormalize(snaps)
    rm = ReducedModel(fom, V)
    names = fom.param_names
    methods: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "rb_galerkin": lambda m: rm.amplitudes(m, "galerkin")[0],
        "rb_lspg": lambda m: rm.amplitudes(m, "lspg")[0],
        "pod_regression": lambda m: bl.pod_regression(fom, mu_train, snaps, m, lo, hi),
    }
    if one_d:
        methods["aaa_t"] = lambda m: bl.direct_aaa(mu_train[:, 0], t_train, m[:, 0])
        methods["pchip_t"] = lambda m: bl.direct_pchip(mu_train[:, 0], t_train, m[:, 0])
    else:
        methods["rbf_t"] = lambda m: bl.direct_rbf(mu_train, t_train, m, lo, hi)

    def fom_T(scan: Scan) -> Callable[[np.ndarray], np.ndarray]:
        return lambda e: np.abs(fom.amplitudes(np.column_stack(
            [e] + [np.full_like(e, scan.heights[p]) for p in names[1:]]))[0]) ** 2

    out: dict[str, dict[str, Any]] = {}
    for name, f in methods.items():
        t0 = time.perf_counter()
        t_hat = f(test_mu)
        dt_test = time.perf_counter() - t0
        err = np.abs(t_hat - t_test)
        res: dict[str, Any] = {
            "test_max_abs_dt": float(err.max()),
            "test_p95_abs_dt": float(np.quantile(err, 0.95)),
            "test_median_abs_dt": float(np.median(err)),
            "test_rmse_T": float(np.sqrt(np.mean((np.abs(t_hat) ** 2 - np.abs(t_test) ** 2) ** 2))),
            "seconds_per_test_point": dt_test / len(test_mu),
            "scans": [],
        }
        for scan in scans:
            mu_s = scan.mu(names)
            t_s = f(mu_s)

            def T_hat(e: np.ndarray, scan: Scan = scan, f: Callable = f) -> np.ndarray:
                m = np.column_stack([e] + [np.full_like(e, scan.heights[p]) for p in names[1:]])
                return np.abs(f(m)) ** 2

            res["scans"].append(scan_metrics(scan, t_s, T_hat, fom_T(scan)))
        sc = res["scans"]
        reson = [r for s in sc for r in s["resonances"]]
        res["scan_rmse_T_mean"] = float(np.mean([s["rmse_T"] for s in sc]))
        res["scan_max_abs_dT"] = float(max(s["max_abs_dT"] for s in sc))
        res["resonances_resolved"] = f"{sum(r['resolved'] for r in reson)}/{len(reson)}"
        res["max_peak_shift_over_gamma"] = float(max(r["peak_shift_over_gamma"] for r in reson)) if reson else None
        widths = [r["fwhm_rel_error"] for r in reson if r["fwhm_rel_error"] is not None]
        res["max_fwhm_rel_error"] = float(max(widths)) if widths else None
        res["fwhm_undefined_or_missing"] = sum(r["fwhm_rel_error"] is None for r in reson)
        res["spurious_peaks_total"] = int(sum(s["spurious_peaks"] for s in sc))
        out[name] = res
    out["_basis"] = {"r": int(V.shape[1]), "rb_offline_seconds": rm.offline_seconds,
                     "singular_values": (sv / sv[0]).tolist()}
    return out


# --------------------------------------------------------------- experiment 1


def aaa_greedy_design(fom: FOM, cand_E: np.ndarray, V0: float, budget: int, n_start: int = 4) -> np.ndarray:
    """AAA's own greedy rule (Phase 3): next point at the minimum of the barycentric denominator.

    Starts from n_start uniform points; uses FOM t values at the chosen points only.
    """
    from qscatter.surrogates.rational import RationalAAA

    lo_E, hi_E = float(cand_E.min()), float(cand_E.max())
    E = list(np.linspace(lo_E, hi_E, n_start))
    t = list(fom.amplitudes(np.column_stack([E, np.full(len(E), V0)]))[0])
    while len(E) < budget:
        r = RationalAAA().fit(np.array(E), np.column_stack([np.real(t), np.imag(t)]))
        d = r.denominator(cand_E)
        d[np.isin(cand_E, E)] = np.inf
        e = float(cand_E[int(np.argmin(d))])
        E.append(e)
        t.append(fom.amplitudes(np.array([[e, V0]]))[0][0])
    return np.column_stack([E, np.full(len(E), V0)])



def _scans(fom: FOM, cfg: dict) -> list[Scan]:
    return [make_scan(fom, h, tuple(cfg["E_range"]), cfg["scan_uniform"], cfg["scan_window"])
            for h in cfg["scan_heights"]]


def run_rb_1d(cfg: dict) -> dict:
    fom = FOM(double_barrier_layers(cfg["barrier_width"], cfg["well_width"], symmetric=True), cfg["h"])
    V0 = cfg["V0"]
    lo_E, hi_E = cfg["E_range"]
    lo, hi = np.array([lo_E, V0]), np.array([hi_E, V0])
    cand_E = np.linspace(lo_E, hi_E, cfg["n_candidates"] + 2)[1:-1]  # interior, offset from tests
    cand = np.column_stack([cand_E, np.full_like(cand_E, V0)])
    scans = _scans(fom, {**cfg, "scan_heights": [{"V0": V0}]})
    test_mu, t_test = scans[0].mu(fom.param_names), scans[0].t_fom
    # FOM accuracy against the exact S-matrix solver on the same scan
    exact = compute_scattering(scans[0].E, fom.layers.potential({"V0": V0}))
    fom_vs_exact = {"max_abs_dt": float(np.max(np.abs(t_test - exact.t))),
                    "rmse_T": l2_rmse(scans[0].E, np.abs(t_test) ** 2 - exact.T)}

    t0 = time.perf_counter()
    greedy = residual_greedy(fom, cand, max(cfg["budgets"]), np.array([0.5 * (lo_E + hi_E), V0]))
    t_greedy = time.perf_counter() - t0
    aaa_greedy = aaa_greedy_design(fom, cand_E, V0, max(cfg["budgets"]))
    rows = []
    for n in cfg["budgets"]:
        designs = {"greedy": greedy.selected[:n],
                   "uniform": np.column_stack([np.linspace(lo_E, hi_E, n), np.full(n, V0)])}
        if n >= 4:
            designs["aaa_greedy"] = aaa_greedy[:n]
        for s in cfg["random_seeds"]:
            designs[f"random_s{s}"] = np.column_stack([np.random.default_rng(s).uniform(lo_E, hi_E, n), np.full(n, V0)])
        for dname, mu_tr in designs.items():
            snaps = fom.solve(mu_tr).T if dname != "greedy" else greedy.snapshots[:, :n]
            t_tr, _ = fom.observables(mu_tr, snaps[0], snaps[-1])
            m = evaluate_methods(fom, mu_tr, snaps, t_tr, test_mu, t_test, scans, lo, hi, one_d=True)
            rows.append({"n": n, "design": dname, **m})
    data = {"provenance": provenance(cfg), "fom": {"nodes": int(fom.nodes.size), **fom_vs_exact},
            "aaa_greedy_selected_E": aaa_greedy[:, 0].tolist(),
            "greedy": {"selected_E": greedy.selected[:, 0].tolist(), "max_indicator": greedy.max_indicator,
                       "seconds_total": t_greedy, "seconds_indicator": greedy.seconds_indicator},
            "test_points": int(test_mu.shape[0]), "poles": [[z.real, z.imag] for z in scans[0].poles],
            "rows": rows}
    return data


# --------------------------------------------------------------- experiment 2


def run_rb_3d(cfg: dict) -> dict:
    fom = FOM(double_barrier_layers(cfg["barrier_width"], cfg["well_width"]), cfg["h"])
    lo, hi = np.array(cfg["lo"], float), np.array(cfg["hi"], float)
    cand = qmc.scale(qmc.Sobol(3, seed=cfg["seed_candidates"]).random(cfg["n_candidates"]), lo, hi)
    test_mu = qmc.scale(qmc.Sobol(3, seed=cfg["seed_test"]).random(cfg["n_test"]), lo, hi)
    t0 = time.perf_counter()
    t_test, _ = fom.amplitudes(test_mu)
    fom_seconds = (time.perf_counter() - t0) / len(test_mu)
    rng = np.random.default_rng(cfg["seed_scans"])
    heights = [{"VL": float(a), "VR": float(b)} for a, b in rng.uniform(lo[1:], hi[1:], (cfg["n_scans"], 2))]
    scans = _scans(fom, {**cfg, "scan_heights": heights})
    nmax = max(cfg["budgets"])
    t0 = time.perf_counter()
    greedy = residual_greedy(fom, cand, nmax, 0.5 * (lo + hi))
    t_greedy = time.perf_counter() - t0
    sobol = qmc.scale(qmc.Sobol(3, seed=cfg["seed_sobol_design"]).random(nmax), lo, hi)
    rows = []
    for n in cfg["budgets"]:
        designs = {"greedy": greedy.selected[:n], "sobol": sobol[:n]}
        for s in cfg["random_seeds"]:
            designs[f"random_s{s}"] = np.random.default_rng(s).uniform(lo, hi, (n, 3))
        side = round(n ** (1 / 3))
        if side**3 == n:
            g = [np.linspace(lo[d], hi[d], side) for d in range(3)]
            designs["tensor"] = np.array(np.meshgrid(*g, indexing="ij")).reshape(3, -1).T
        for dname, mu_tr in designs.items():
            snaps = greedy.snapshots[:, :n] if dname == "greedy" else fom.solve(mu_tr).T
            t_tr, _ = fom.observables(mu_tr, snaps[0], snaps[-1])
            m = evaluate_methods(fom, mu_tr, snaps, t_tr, test_mu, t_test, scans, lo, hi, one_d=False)
            rows.append({"n": n, "design": dname, **m})
    # residual indicator vs true error (held-out test points; greedy bases)
    eff = []
    sub = test_mu[: cfg["n_effectivity"]]
    t_sub = t_test[: cfg["n_effectivity"]]
    psi_sub = fom.solve(sub)
    inv = np.array([bl.inverse_norm(fom, m) for m in sub])
    for n in cfg["effectivity_budgets"]:
        rm = ReducedModel(fom, orthonormalize(greedy.snapshots[:, :n])[0])
        c = rm.coefficients(sub)
        psi_hat = c @ rm.V.T
        res_abs = rm.residual_norm(sub, c)
        err_psi = np.linalg.norm(psi_hat - psi_sub, axis=1)
        t_hat, _ = fom.observables(sub, psi_hat[:, 0], psi_hat[:, -1])
        err_t = np.abs(t_hat - t_sub)
        bound = inv * res_abs
        eff.append({
            "n": n,
            "spearman_residual_vs_err_t": float(spearmanr(res_abs, err_t).statistic),
            "spearman_residual_vs_err_psi": float(spearmanr(res_abs, err_psi).statistic),
            "bound_holds_fraction": float(np.mean(bound >= err_psi * (1 - 1e-9))),
            "bound_effectivity_median": float(np.median(bound / err_psi)),
            "bound_effectivity_max": float(np.max(bound / err_psi)),
            "residual": res_abs.tolist(), "err_psi": err_psi.tolist(), "err_t": err_t.tolist(),
        })
    return {"provenance": provenance(cfg), "fom": {"nodes": int(fom.nodes.size), "seconds_per_solve": fom_seconds},
            "greedy": {"selected": greedy.selected.tolist(), "max_indicator": greedy.max_indicator,
                       "seconds_total": t_greedy, "seconds_indicator": greedy.seconds_indicator,
                       "seconds_fom": greedy.seconds_fom},
            "scans": [{"heights": s.heights, "n_points": int(s.E.size),
                       "poles": [[z.real, z.imag] for z in s.poles]} for s in scans],
            "inverse_norm": inv.tolist(), "effectivity": eff, "rows": rows}


# --------------------------------------------------------------- experiment 3


def run_nwidth(cfg: dict) -> dict:
    """Singular values of dense snapshot sets in E for several geometries (1D in E)."""
    out = []
    for case in cfg["cases"]:
        lay = double_barrier_layers(case["barrier_width"], case["well_width"], symmetric=True)
        fom = FOM(lay, cfg["h"])
        lo_E, hi_E = cfg["E_range"]
        pot = lay.potential({"V0": case["V0"]})
        poles = [z for z in find_poles(pot, (lo_E, hi_E)) if lo_E <= z.real <= hi_E]
        E = [np.linspace(lo_E, hi_E, cfg["n_uniform"])]
        for z in poles:  # make sure every resonance core is in the snapshot set
            G = -2 * z.imag
            E.append(np.linspace(max(lo_E, z.real - 5 * G), min(hi_E, z.real + 5 * G), cfg["n_window"]))
        Es = np.unique(np.concatenate(E))
        S = fom.solve(np.column_stack([Es, np.full_like(Es, case["V0"])])).T
        s = np.linalg.svd(S, compute_uv=False)
        s = s / s[0]
        k_lo, k_hi = np.sqrt(lo_E), np.sqrt(hi_E)
        out.append({
            **case,
            "n_snapshots": int(Es.size),
            "n_poles_in_range": len(poles),
            "gammas": [-2 * z.imag for z in poles],
            "well_phase_count": case["well_width"] * (k_hi - k_lo) / np.pi,
            "rank_1e-6": int(np.sum(s > 1e-6)),
            "rank_1e-10": int(np.sum(s > 1e-10)),
            "singular_values": s[:80].tolist(),
        })
    return {"provenance": provenance(cfg), "cases": out}
