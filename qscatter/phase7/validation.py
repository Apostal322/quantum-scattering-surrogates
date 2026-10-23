"""Experiment A: independent physics validation and convergence of the Phase 7 FEM model.

All references are independent of the FEM discretisation:
- the DOP853 ODE integrator (`qscatter.physics.ode_solver.solve_ode_smooth`), integrated backward
  from the transmitted wave on the same truncated domain [-X, X];
- for poles (complex E), Richardson extrapolation of FEM poles from the two finest meshes,
  since the ODE solver works on the real axis only.

Tolerances are taken from these studies (see docs/PHASE7_NUMERICAL_VALIDATION.md), not fixed a priori.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.signal import find_peaks

from qscatter.phase7.fom import gaussian_double_barrier, gaussian_fom, truncation_halfwidth
from qscatter.phase7.resonances import find_poles
from qscatter.physics.ode_solver import solve_ode_smooth


def ode_amplitudes(E: np.ndarray, heights, s: float, d: float, X: float, rtol: float) -> tuple[np.ndarray, np.ndarray]:
    V = gaussian_double_barrier(heights[0], heights[1], s, s, d)
    res = [solve_ode_smooth(float(e), V, -X, X, rtol=rtol, atol=1e-15) for e in E]
    return np.array([o.t for o in res]), np.array([o.r for o in res])


def _mu(E: np.ndarray, heights) -> np.ndarray:
    return np.column_stack([E] + [np.full_like(E, v) for v in heights])


def run_validation(cfg: dict) -> dict[str, Any]:
    s, d = cfg["s"], cfg["d"]
    out: dict[str, Any] = {"configs": []}
    E_chk = np.linspace(cfg["E_range"][0] + 0.05, cfg["E_range"][1], cfg["n_check_energies"])
    for heights in cfg["heights"]:
        rec: dict[str, Any] = {"heights": heights}
        X = truncation_halfwidth(d, s, cfg["rel_cut"])
        # A1 ODE self-consistency
        t_ref, r_ref = ode_amplitudes(E_chk, heights, s, d, X, cfg["ode_rtol"])
        rec["ode_tolerance"] = {
            str(rt): float(np.max(np.abs(ode_amplitudes(E_chk, heights, s, d, X, rt)[0] - t_ref)))
            for rt in cfg["ode_rtol_check"]
        }
        rec["ode_flux_defect"] = float(np.max(np.abs(np.abs(t_ref) ** 2 + np.abs(r_ref) ** 2 - 1)))
        # A2 domain truncation (ODE, so the mesh plays no role)
        X_big = truncation_halfwidth(d, s, min(cfg["rel_cuts"]))
        t_big, _ = ode_amplitudes(E_chk, heights, s, d, X_big, cfg["ode_rtol"])
        rec["truncation"] = {
            str(rc): float(np.max(np.abs(ode_amplitudes(E_chk, heights, s, d, truncation_halfwidth(d, s, rc), cfg["ode_rtol"])[0] - t_big)))
            for rc in cfg["rel_cuts"]
        }
        # A3 mesh convergence vs ODE (same X)
        mesh = []
        poles_by_h = {}
        for h in cfg["meshes"]:
            fom = gaussian_fom(s, s, d, X, h)
            t0 = time.perf_counter()
            t, r = fom.amplitudes(_mu(E_chk, heights))
            dt = (time.perf_counter() - t0) / E_chk.size
            T, T_ref = np.abs(t) ** 2, np.abs(t_ref) ** 2
            poles = find_poles(fom, tuple(heights), tuple(cfg["E_range"]))
            poles_by_h[h] = poles
            mesh.append({
                "h": h, "nodes": int(fom.nodes.size), "seconds_per_solve": dt,
                "max_abs_dt": float(np.max(np.abs(t - t_ref))),
                "max_abs_dr": float(np.max(np.abs(r - r_ref))),
                "max_abs_dT": float(np.max(np.abs(T - T_ref))),
                "mae_T": float(np.mean(np.abs(T - T_ref))),
                "flux_defect": float(np.max(np.abs(np.abs(t) ** 2 + np.abs(r) ** 2 - 1))),
                "poles": [[z.real, z.imag] for z in poles],
            })
        # pole convergence: Richardson from the two finest meshes (p = 2), matched by nearest
        hs = sorted(cfg["meshes"])
        fine, finer = poles_by_h[hs[1]], poles_by_h[hs[0]]
        ref_poles = []
        for z1 in finer:
            z2 = min(fine, key=lambda z: abs(z - z1)) if fine else None
            ref_poles.append(z1 + (z1 - z2) / 3 if z2 is not None and abs(z2 - z1) < 1e-3 else z1)
        for m in mesh:
            errs = []
            for zr in ref_poles:
                ps = [complex(*p) for p in m["poles"]]
                if not ps:
                    errs.append(None)
                    continue
                z = min(ps, key=lambda q: abs(q - zr))
                G = -2 * zr.imag
                errs.append({"E_r": zr.real, "gamma": G, "abs_err_Er": abs(z.real - zr.real),
                             "rel_err_gamma": abs(-2 * z.imag - G) / G,
                             "err_Er_over_gamma": abs(z.real - zr.real) / G})
            m["pole_errors"] = errs
        rec["mesh"] = mesh
        rec["reference_poles"] = [[z.real, z.imag] for z in ref_poles]
        # A4 energy-grid convergence: peaks found by find_peaks on uniform grids (FEM at h_ref)
        fom = gaussian_fom(s, s, d, X, cfg["h_ref"])
        grid = []
        for n in cfg["energy_grids"]:
            E = np.linspace(*cfg["E_range"], n)
            T = np.abs(fom.amplitudes(_mu(E, heights))[0]) ** 2
            idx, _ = find_peaks(T, prominence=0.05)
            matched = [any(abs(E[i] - zr.real) < 2 * (-2 * zr.imag) for i in idx) for zr in ref_poles]
            grid.append({"n": n, "spacing": float(E[1] - E[0]), "peaks": int(idx.size),
                         "poles_with_a_peak": int(sum(matched)), "max_T_found": float(T.max())})
        rec["energy_grid"] = grid
        # A5 stability of poles under domain change (artifact check)
        fom_X2 = gaussian_fom(s, s, d, truncation_halfwidth(d, s, cfg["rel_cut"] * 1e-2), cfg["h_ref"])
        poles_X2 = find_poles(fom_X2, tuple(heights), tuple(cfg["E_range"]))
        rec["pole_shift_domain_change"] = [
            min((abs(z - complex(*p)) for z in poles_X2), default=None) for p in rec["reference_poles"]
        ]
        out["configs"].append(rec)
    return out
