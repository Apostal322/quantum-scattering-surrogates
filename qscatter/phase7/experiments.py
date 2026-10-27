"""Phase 7 experiments B (generalisation), C (fair comparison) and E (residual reliability).

Experiment D (basis size) and F (robustness) live in `basis.py` and `robustness.py`.

Problem: smooth double-Gaussian barrier (qscatter.phase7.fom), parameters mu = (E, V_L, V_R),
fixed widths s and separation d (they enter non-affinely). Accuracy is measured against the FEM
model at the mesh chosen in Experiment A; the FEM's own error against the ODE reference is
reported separately (the error floor).

Protocol (qscatter.phase7.protocol): separate seeds for greedy candidates, designs, validation and
tests; hyperparameters on validation only; data-driven baselines on the pre-declared design.
"""
from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.stats import spearmanr

from qscatter.phase6 import baselines as bl6
from qscatter.phase6.experiments import Scan, scan_metrics
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy
from qscatter.phase7 import mlp
from qscatter.phase7.fom import AffineFOM, gaussian_fom, rectangular_fom_on_mesh
from qscatter.phase7.protocol import make_splits, sobol
from qscatter.phase7.resonances import find_poles

# ------------------------------------------------------------------ helpers


def make_scan(fom: AffineFOM, heights, E_range, n_uniform: int, n_window: int) -> Scan:
    poles = find_poles(fom, tuple(heights), tuple(E_range))
    lo, hi = E_range
    grids = [np.linspace(lo, hi, n_uniform)]
    for z in poles:
        G = -2 * z.imag
        grids.append(np.linspace(max(lo, z.real - 20 * G), min(hi, z.real + 20 * G), n_window))
    E = np.unique(np.concatenate(grids))
    mu = np.column_stack([E, np.full_like(E, heights[0]), np.full_like(E, heights[1])])
    t, _ = fom.amplitudes(mu)
    return Scan({"VL": heights[0], "VR": heights[1]}, E, t, poles)


def point_metrics(t_hat: np.ndarray, t: np.ndarray) -> dict[str, float]:
    T_hat, T = np.abs(t_hat) ** 2, np.abs(t) ** 2
    return {
        "mae_T": float(np.mean(np.abs(T_hat - T))),
        "rmse_T": float(np.sqrt(np.mean((T_hat - T) ** 2))),
        "max_abs_dT": float(np.max(np.abs(T_hat - T))),
        "max_abs_dt": float(np.max(np.abs(t_hat - t))),
        "median_abs_dt": float(np.median(np.abs(t_hat - t))),
    }


def scan_summary(scans: list[Scan], f: Callable[[np.ndarray], np.ndarray], fom: AffineFOM) -> dict[str, Any]:
    names = ["E", "VL", "VR"]
    per = []
    for sc in scans:
        mu = sc.mu(names)
        t_hat = f(mu)

        def T_hat(e, sc=sc):
            return np.abs(f(np.column_stack([e, np.full_like(e, sc.heights["VL"]), np.full_like(e, sc.heights["VR"])]))) ** 2

        def T_fom(e, sc=sc):
            return np.abs(fom.amplitudes(np.column_stack([e, np.full_like(e, sc.heights["VL"]), np.full_like(e, sc.heights["VR"])]))[0]) ** 2

        m = scan_metrics(sc, t_hat, T_hat, T_fom)
        T, Th = np.abs(sc.t_fom) ** 2, np.abs(t_hat) ** 2
        m["mae_T"] = float(np.trapezoid(np.abs(Th - T), sc.E) / (sc.E[-1] - sc.E[0]))
        per.append(m)
    res = [r for m in per for r in m["resonances"]]
    widths = [r["fwhm_rel_error"] for r in res if r["fwhm_rel_error"] is not None]
    return {
        "rmse_T": float(np.mean([m["rmse_T"] for m in per])),
        "mae_T": float(np.mean([m["mae_T"] for m in per])),
        "max_abs_dT": float(max(m["max_abs_dT"] for m in per)),
        "max_abs_dt": float(max(m["max_abs_dt"] for m in per)),
        "resonances": len(res),
        "recall": float(np.mean([r["resolved"] for r in res])) if res else None,
        "max_peak_shift_over_gamma": float(max(r["peak_shift_over_gamma"] for r in res)) if res else None,
        "max_fwhm_rel_error": float(max(widths)) if widths else None,
        "fwhm_undefined_or_missing": int(sum(r["fwhm_rel_error"] is None for r in res)),
        "spurious_peaks": int(sum(m["spurious_peaks"] for m in per)),
        "spurious_peaks_fom": int(sum(m["spurious_peaks_fom"] for m in per)),
    }


def rbf_fit(X, Y, kernel: str):
    # the quintic kernel is conditionally positive definite of order 3 and needs a degree-2 polynomial
    degree = 2 if kernel == "quintic" else 1
    return RBFInterpolator(X, np.column_stack([Y.real, Y.imag]), kernel=kernel, degree=degree)


def rbf_predict(model, Xq, m: int):
    Z = model(Xq)
    return Z[:, :m] + 1j * Z[:, m:]


def greedy_from(fom: AffineFOM, cand: np.ndarray, extra: int, init_snaps: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Residual greedy that starts from an existing basis (used for basis enrichment in B3)."""
    snaps = [s for s in init_snaps.T]
    sel = []
    t0 = time.perf_counter()
    for _ in range(extra):
        V, _ = orthonormalize(np.column_stack(snaps))
        _, _, res = ReducedModel(fom, V).amplitudes(cand)
        j = int(np.argmax(res))
        sel.append(cand[j])
        snaps.append(fom.solve(cand[j]))
    return np.column_stack(snaps), np.array(sel), time.perf_counter() - t0


# --------------------------------------------------------- experiment B / C / E


def run_generalization(cfg: dict) -> dict[str, Any]:
    s, d, X, h = cfg["s"], cfg["d"], cfg["X"], cfg["h"]
    fom = gaussian_fom(s, s, d, X, h)
    sp = make_splits(cfg)
    lo, hi = np.array(cfg["train_lo"]), np.array(cfg["train_hi"])
    E_range = (lo[0], hi[0])
    t0 = time.perf_counter()
    t_val = fom.amplitudes(sp.validation)[0]
    t_test = fom.amplitudes(sp.test)[0]
    t_ood = fom.amplitudes(sp.test_ood)[0]
    sc = cfg["scan"]
    scans_test = [make_scan(fom, hh, E_range, sc["n_uniform"], sc["n_window"]) for hh in sp.scan_heights_test]
    scans_ood = [make_scan(fom, hh, E_range, sc["n_uniform"], sc["n_window"]) for hh in sp.scan_heights_ood]
    truth_seconds = time.perf_counter() - t0

    def evaluate(f: Callable[[np.ndarray], np.ndarray]) -> dict[str, Any]:
        t1 = time.perf_counter()
        th = f(sp.test)
        online = (time.perf_counter() - t1) / len(sp.test)
        return {
            "test": point_metrics(th, t_test),
            "test_ood": point_metrics(f(sp.test_ood), t_ood),
            "scans_test": scan_summary(scans_test, f, fom),
            "scans_ood": scan_summary(scans_ood, f, fom),
            "seconds_per_query_batched": online,
        }

    nmax = max(cfg["budgets"])
    tg = time.perf_counter()
    greedy = residual_greedy(fom, sp.candidates, nmax, 0.5 * (lo + hi))
    greedy_seconds = time.perf_counter() - tg
    designs_all = {"greedy": greedy.selected,
                   "sobol": sobol(nmax, lo, hi, cfg["seeds"]["design_sobol"])}
    for r in cfg["seeds"]["design_random"]:
        designs_all[f"random_{r}"] = np.random.default_rng(r).uniform(lo, hi, (nmax, 3))

    # RBF kernel chosen on validation, at the declared design and the median budget (frozen after)
    n_sel = cfg["budgets"][len(cfg["budgets"]) // 2]
    Xd = designs_all[cfg["baseline_design"]][:n_sel]
    td = fom.amplitudes(Xd)[0]
    unit = lambda A: (np.atleast_2d(A) - lo) / (hi - lo)  # noqa: E731
    kernel_val = {}
    for k in cfg["rbf_kernels"]:
        pred = rbf_predict(rbf_fit(unit(Xd), td[:, None], k), unit(sp.validation), 1)[:, 0]
        kernel_val[k] = float(np.sqrt(np.mean(np.abs(pred - t_val) ** 2)))
    kernel = min(kernel_val, key=kernel_val.get)

    rows = []
    for n in cfg["budgets"]:
        for dname, Xall in designs_all.items():
            mu_tr = Xall[:n]
            snaps = greedy.snapshots[:, :n] if dname == "greedy" else fom.solve(mu_tr).T
            t_tr, _ = fom.observables(mu_tr, snaps[0], snaps[-1])
            tb = time.perf_counter()
            V, sv = orthonormalize(snaps)
            rm = ReducedModel(fom, V)
            basis_seconds = time.perf_counter() - tb
            row: dict[str, Any] = {"n": n, "design": dname, "r": int(V.shape[1]),
                                   "basis_and_assembly_seconds": basis_seconds,
                                   "rb_galerkin": evaluate(lambda m: rm.amplitudes(m, residual=False)[0])}
            if dname == "greedy":
                row["rb_lspg"] = evaluate(lambda m: rm.amplitudes(m, "lspg", residual=False)[0])
            if dname == cfg["baseline_design"]:
                tf = time.perf_counter()
                rbf = rbf_fit(unit(mu_tr), t_tr[:, None], kernel)
                fit_rbf = time.perf_counter() - tf
                row["rbf_t"] = evaluate(lambda m: rbf_predict(rbf, unit(m), 1)[:, 0])
                row["rbf_t"]["fit_seconds"] = fit_rbf
                row["pod_regression"] = evaluate(lambda m: bl6.pod_regression(fom, mu_tr, snaps, m, lo, hi))
            rows.append(row)

    # MLP on its own (larger) Sobol designs of scalar t evaluations; selection on validation only
    mlp_rows = []
    for n in cfg["mlp_budgets"]:
        Xm = sobol(n, lo, hi, cfg["seeds"]["design_mlp"])
        tm = fom.amplitudes(Xm)[0]
        pred, choice, fit_s = mlp.select_and_fit(Xm, tm, sp.validation, t_val, lo, hi, cfg["mlp_grid"], seed=0)
        r = evaluate(pred)
        r.update({"n": n, "choice": choice, "fit_seconds": fit_s})
        mlp_rows.append(r)

    # ---------------- B3: rectangular -> Gaussian transfer on the same mesh
    rect = rectangular_fom_on_mesh(cfg["rect_width"], d, X, h)
    rlo, rhi = np.array(cfg["rect_lo"]), np.array(cfg["rect_hi"])
    rcand = sobol(cfg["n_candidates"], rlo, rhi, cfg["seeds"]["rect_candidates"])
    rg = residual_greedy(rect, rcand, max(cfg["rect_budgets"]), 0.5 * (rlo + rhi))
    transfer = []
    for nr in cfg["rect_budgets"]:
        Vr, _ = orthonormalize(rg.snapshots[:, :nr])
        rm = ReducedModel(fom, Vr)  # Gaussian operator projected on the rectangular basis
        transfer.append({"n_rect": nr, "n_gauss": 0, "r": int(Vr.shape[1]),
                         "result": evaluate(lambda m: rm.amplitudes(m, residual=False)[0])})
    nr0 = cfg["rect_enrich_base"]
    for m_extra in cfg["enrich_budgets"]:
        S, _, _ = greedy_from(fom, sp.candidates, m_extra, rg.snapshots[:, :nr0])
        V, _ = orthonormalize(S)
        rm = ReducedModel(fom, V)
        transfer.append({"n_rect": nr0, "n_gauss": m_extra, "r": int(V.shape[1]),
                         "result": evaluate(lambda mm: rm.amplitudes(mm, residual=False)[0])})

    # ---------------- E: residual reliability with the residual/sigma_min a-posteriori bound (greedy bases)
    eff = []
    sub = sp.test[: cfg["n_effectivity"]]
    psi_sub = fom.solve(sub)
    smin = np.array([fom.sigma_min(m, certified=True) for m in sub])
    inv_lower = np.array([bl6.inverse_norm(fom, m) for m in sub])
    # resonance neighbourhoods: points within 5 Gamma of a pole on the test scans
    near_mu = []
    for sc_ in scans_test:
        for z in sc_.poles:
            G = -2 * z.imag
            for e in z.real + G * np.array([-2, -1, -0.5, 0, 0.5, 1, 2]):
                if E_range[0] < e < E_range[1]:
                    near_mu.append([e, sc_.heights["VL"], sc_.heights["VR"]])
    near_mu = np.array(near_mu)
    psi_near = fom.solve(near_mu)
    smin_near = np.array([fom.sigma_min(m, certified=True) for m in near_mu])
    for n in cfg["effectivity_budgets"]:
        rm = ReducedModel(fom, orthonormalize(greedy.snapshots[:, :n])[0])
        blk = {}
        for label, mus, psis, sm in (("random", sub, psi_sub, smin), ("near_resonance", near_mu, psi_near, smin_near)):
            c = rm.coefficients(mus)
            ph = c @ rm.V.T
            res = rm.residual_norm(mus, c)
            e_psi = np.linalg.norm(ph - psis, axis=1)
            th, _ = fom.observables(mus, ph[:, 0], ph[:, -1])
            tt, _ = fom.observables(mus, psis[:, 0], psis[:, -1])
            e_t = np.abs(th - tt)
            e_T = np.abs(np.abs(th) ** 2 - np.abs(tt) ** 2)
            bound = res / sm
            blk[label] = {
                "n_points": int(len(mus)),
                "spearman_res_vs_err_psi": float(spearmanr(res, e_psi).statistic),
                "spearman_res_vs_err_t": float(spearmanr(res, e_t).statistic),
                "spearman_res_vs_err_T": float(spearmanr(res, e_T).statistic),
                "certified_bound_holds_psi": float(np.mean(bound >= e_psi * (1 - 1e-8))),
                "certified_bound_holds_t": float(np.mean(bound >= e_t * (1 - 1e-8))),
                "bound_over_err_psi_median": float(np.median(bound / np.maximum(e_psi, 1e-300))),
                "bound_over_err_t_median": float(np.median(bound / np.maximum(e_t, 1e-300))),
                "residual": res.tolist(), "err_psi": e_psi.tolist(), "err_t": e_t.tolist(), "err_T": e_T.tolist(),
                "sigma_min": sm.tolist(),
            }
        eff.append({"n": n, **blk})
    return {
        "manifest": sp.manifest,
        "fom": {"nodes": int(fom.nodes.size), "h": h, "X": X},
        "truth_seconds": truth_seconds,
        "greedy": {"selected": greedy.selected.tolist(), "max_indicator": greedy.max_indicator,
                   "seconds_total": greedy_seconds, "seconds_indicator": greedy.seconds_indicator},
        "rbf_kernel_validation": kernel_val, "rbf_kernel": kernel,
        "rows": rows, "mlp": mlp_rows, "transfer": transfer,
        "scans": {"test": [{"heights": x.heights, "poles": [[z.real, z.imag] for z in x.poles]} for x in scans_test],
                  "ood": [{"heights": x.heights, "poles": [[z.real, z.imag] for z in x.poles]} for x in scans_ood]},
        "power_iteration_vs_exact_inverse_norm": {
            "max_ratio_lower_over_exact": float(np.max(inv_lower * smin)),
            "min_ratio_lower_over_exact": float(np.min(inv_lower * smin)),
        },
        "effectivity": eff,
    }


RUNNERS: dict[str, Callable[[dict], dict]] = {"generalization": run_generalization}
