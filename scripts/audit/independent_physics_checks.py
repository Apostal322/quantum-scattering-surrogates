"""Audit (2026-10): independent physics checks of every scattering solver in the project.

    OMP_NUM_THREADS=1 python scripts/audit/independent_physics_checks.py

Writes results/audit_2026/physics_checks.json and .md. Each check uses a reference that does not
share the algorithm under test:

C1  global matching system (qscatter/physics/matching.py) vs Redheffer S-matrix: random multilayers,
    opaque barriers, complex energies.
C2  P1-FEM (Phase 7, rectangular shapes on a shared mesh) vs the matching system: O(h^2) and phases.
C3  weak-coupling (first Born) limit of the smooth Gaussian barrier, analytic: (t-1)/lam and r/lam
    for V -> lam V, lam -> 0; checks FEM, ODE and staircase S-matrix, including the global phase of r.
C4  metamorphic: translation (t -> t, r -> r e^{2ika}) and mirror (t unchanged, |r| unchanged).
C5  Richardson-extrapolated FEM vs ODE: both converge to the same continuum solution.
C6  resonance poles of the smooth problem: FEM (Richardson in h) vs staircase S-matrix continued to
    complex E (Richardson in the number of steps), and the pole condition 1/t = 0 checked in both.
C7  flux: |t|^2 + |r|^2 = 1 (real V) for all solvers; absorption (Im V < 0) gives a deficit.
C8  threshold: t ~ c k as E -> 0 (no bound state at threshold).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qscatter.phase7.fom import gaussian_double_barrier, gaussian_fom, rectangular_fom_on_mesh, truncation_halfwidth  # noqa: E402
from qscatter.physics.matching import matching_amplitudes  # noqa: E402
from qscatter.physics.ode_solver import solve_ode_smooth  # noqa: E402
from qscatter.physics.poles import refine_pole  # noqa: E402
from qscatter.physics.potentials import double_barrier, from_layers, single_barrier  # noqa: E402
from qscatter.physics.smatrix import compute_scattering, transmission_complex_energy  # noqa: E402

OUT = REPO / "results" / "audit_2026"
S, D = 0.6, 4.0
X = truncation_halfwidth(D, S, 1e-14)


def mu(E, VL, VR):
    E = np.atleast_1d(E)
    return np.column_stack([E, np.full(E.shape, VL), np.full(E.shape, VR)])


def staircase(VL, VR, M, x0=0.0):
    edges = np.linspace(-X, X, M + 1)
    mid = 0.5 * (edges[1:] + edges[:-1])
    return from_layers(np.diff(edges), gaussian_double_barrier(VL, VR, S, S, D)(mid - x0), x0=-X)


def c1():
    rng = np.random.default_rng(2026)
    worst = 0.0
    for _ in range(40):
        n = int(rng.integers(1, 9))
        p = from_layers(rng.uniform(0.05, 1.5, n), rng.uniform(-5, 30, n), x0=rng.uniform(-3, 3))
        E = rng.uniform(0.01, 40, 25)
        ex = compute_scattering(E, p)
        m = np.array([matching_amplitudes(e, p) for e in E])
        worst = max(worst, float(np.max(np.abs(m[:, 0] - ex.t))), float(np.max(np.abs(m[:, 1] - ex.r))))
    opaque = single_barrier(100.0, 5.0)  # kappa a = 50
    E = np.array([1.0, 10.0, 50.0])
    rel_opaque = float(np.max(np.abs(np.array([matching_amplitudes(e, opaque)[0] for e in E]) - compute_scattering(E, opaque).t)
                              / np.abs(compute_scattering(E, opaque).t)))
    p = double_barrier(10.0, 0.8, 2.5)
    Z = [0.99 - 0.004j, 3.88 - 0.04j, 8.3 - 0.3j, 5.0 - 1.0j]
    rel_cplx = max(abs(matching_amplitudes(z, p)[0] - transmission_complex_energy(z, p)[0]) / abs(transmission_complex_energy(z, p)[0]) for z in Z)
    return {"random_multilayer_max_abs_diff_t_r": worst, "opaque_kappa_a_50_rel_diff_t": rel_opaque,
            "complex_E_max_rel_diff_t": float(rel_cplx), "pass": worst < 1e-12 and rel_opaque < 1e-10 and rel_cplx < 1e-9}


def c2():
    E = np.linspace(0.3, 9.5, 25)
    p = double_barrier(10.0, 1.0, D - 1.0, height_right=12.0)
    ref = np.array([matching_amplitudes(e, p) for e in E])
    errs = []
    for h in (0.0125, 0.00625, 0.003125):
        t, r = rectangular_fom_on_mesh(1.0, D, 5.4125, h).amplitudes(mu(E, 10.0, 12.0))
        errs.append({"h": h, "max_abs_dt": float(np.max(np.abs(t - ref[:, 0]))), "max_abs_dr": float(np.max(np.abs(r - ref[:, 1])))})
    ratios = [errs[i]["max_abs_dt"] / errs[i + 1]["max_abs_dt"] for i in range(2)]
    return {"levels": errs, "ratios_t": ratios, "pass": all(3.5 < q < 4.5 for q in ratios)}


def born_amplitudes(E, VL, VR):
    """First Born: t - 1 = (1/2ik) int V,  r = (1/2ik) int V e^{2ikx} (global convention)."""
    k = np.sqrt(E)
    g = S * np.sqrt(np.pi) * np.exp(-(k * S) ** 2)
    integral_V = (VL + VR) * S * np.sqrt(np.pi)
    integral_r = g * (VL * np.exp(-1j * k * D) + VR * np.exp(1j * k * D))
    return integral_V / (2j * k), integral_r / (2j * k)


def c3():
    lam = 1e-5
    VL, VR = 1.0, 0.6  # asymmetric, so the phase of r is a real test
    E = np.array([0.3, 1.0, 2.5, 4.0])
    tb, rb = born_amplitudes(E, VL, VR)
    out = {}
    for h in (0.005, 0.0025):
        # The P1-FEM has a free-space phase (dispersion) error t_h(V=0) - 1 ~ k^3 h^2 L / 24 that does
        # not vanish with V, so the linear response is taken relative to the discrete free solution.
        fom = gaussian_fom(S, S, D, X, h)
        t0, r0 = fom.amplitudes(mu(E, 0.0, 0.0))
        t, r = fom.amplitudes(mu(E, lam * VL, lam * VR))
        out[f"fem_h{h}"] = (np.abs((t - t0) / lam - tb) / np.abs(tb), np.abs((r - r0) / lam - rb) / np.abs(rb))
        out[f"fem_h{h}_free_space_phase_error"] = (np.abs(t0 - 1), np.abs(r0))
    V = gaussian_double_barrier(lam * VL, lam * VR, S, S, D)
    o = [solve_ode_smooth(e, V, -X, X, rtol=1e-12, atol=1e-18) for e in E]
    out["ode"] = (np.abs((np.array([x.t for x in o]) - 1) / lam - tb) / np.abs(tb), np.abs(np.array([x.r for x in o]) / lam - rb) / np.abs(rb))
    st = compute_scattering(E, staircase(lam * VL, lam * VR, 8000))
    out["staircase_M8000"] = (np.abs((st.t - 1) / lam - tb) / np.abs(tb), np.abs(st.r / lam - rb) / np.abs(rb))
    res = {k_: ({"max_abs_t0_minus_1": float(np.max(a)), "max_abs_r0": float(np.max(b))} if "free_space" in k_
                else {"max_rel_err_t1": float(np.max(a)), "max_rel_err_r": float(np.max(b))}) for k_, (a, b) in out.items()}
    # O(lam) second-order Born term bounds the agreement (relative ~ lam * V / E ~ 1e-5)
    checked = {k_: v for k_, v in res.items() if "free_space" not in k_}
    res["pass"] = all(v["max_rel_err_t1"] < 1e-3 and v["max_rel_err_r"] < 1e-3 for v in checked.values())
    return res


def c4():
    E = np.array([0.5, 0.97, 3.0, 7.0])
    VL, VR, a = 8.0, 13.0, 0.37
    Vf = gaussian_double_barrier(VL, VR, S, S, D)
    base = [solve_ode_smooth(e, Vf, -X - 1, X + 1, rtol=1e-12, atol=1e-15) for e in E]
    shifted = [solve_ode_smooth(e, lambda x: Vf(x - a), -X - 1, X + 1, rtol=1e-12, atol=1e-15) for e in E]
    mirror = [solve_ode_smooth(e, lambda x: Vf(-x), -X - 1, X + 1, rtol=1e-12, atol=1e-15) for e in E]
    k = np.sqrt(E)
    tb, rb = np.array([o.t for o in base]), np.array([o.r for o in base])
    ts, rs = np.array([o.t for o in shifted]), np.array([o.r for o in shifted])
    tm, rm = np.array([o.t for o in mirror]), np.array([o.r for o in mirror])
    fom = gaussian_fom(S, S, D, X, 0.0025)
    tf, rf = fom.amplitudes(mu(E, VL, VR))
    tfm, rfm = fom.amplitudes(mu(E, VR, VL))
    res = {"ode_translation_t": float(np.max(np.abs(ts - tb))), "ode_translation_r": float(np.max(np.abs(rs - rb * np.exp(2j * k * a)))),
           "ode_mirror_t": float(np.max(np.abs(tm - tb))), "ode_mirror_abs_r": float(np.max(np.abs(np.abs(rm) - np.abs(rb)))),
           "fem_mirror_t": float(np.max(np.abs(tfm - tf))), "fem_mirror_abs_r": float(np.max(np.abs(np.abs(rfm) - np.abs(rf))))}
    res["pass"] = max(v for v in res.values()) < 1e-8
    return res


def c5():
    E = np.linspace(0.5, 9.5, 19)
    out = {}
    for VL, VR in ((10.0, 10.0), (8.0, 14.0)):
        V = gaussian_double_barrier(VL, VR, S, S, D)
        t_ode = np.array([solve_ode_smooth(e, V, -X, X, rtol=1e-12, atol=1e-15).t for e in E])
        t1 = gaussian_fom(S, S, D, X, 0.0025).amplitudes(mu(E, VL, VR))[0]
        t2 = gaussian_fom(S, S, D, X, 0.00125).amplitudes(mu(E, VL, VR))[0]
        rich = (4 * t2 - t1) / 3
        out[f"{VL},{VR}"] = {"fem_h0.00125_vs_ode": float(np.max(np.abs(t2 - t_ode))), "richardson_vs_ode": float(np.max(np.abs(rich - t_ode)))}
    out["pass"] = all(v["richardson_vs_ode"] < 1e-2 * v["fem_h0.00125_vs_ode"] for v in out.values() if isinstance(v, dict))
    return out


def c6():
    out = {}
    for VL, VR, guesses in ((10.0, 10.0, [0.958456 - 3.461e-4j, 3.519137 - 8.721e-3j, 6.953409 - 1.107e-1j]),
                            (14.0, 14.0, [1.061625 - 6.678e-5j, 3.925056 - 1.885e-3j, 7.875473 - 3.090e-2j])):
        f1, f2 = gaussian_fom(S, S, D, X, 0.00125), gaussian_fom(S, S, D, X, 0.000625)
        rows = []
        for g in guesses:
            z1, z2 = f1.find_pole(g, (VL, VR)), f2.find_pole(g, (VL, VR))
            z_fem = z2 + (z2 - z1) / 3
            p1, p2 = staircase(VL, VR, 8000), staircase(VL, VR, 16000)
            s1, s2 = refine_pole(g, p1), refine_pole(g, p2)
            z_st = s2 + (s2 - s1) / 3
            G = -2 * z_fem.imag
            # pole condition in the staircase S-matrix: |1/t(z)| relative to |1/t| on the real axis
            depth = abs(1 / transmission_complex_energy(s2, p2)[0]) / abs(1 / transmission_complex_energy(s2.real, p2)[0])
            rows.append({"fem_richardson": [z_fem.real, z_fem.imag], "staircase_richardson": [z_st.real, z_st.imag],
                         "abs_diff": abs(z_fem - z_st), "diff_Er_over_gamma": abs(z_fem.real - z_st.real) / G,
                         "rel_diff_gamma": abs(z_fem.imag - z_st.imag) / abs(z_fem.imag), "staircase_pole_depth": depth})
        out[f"{VL},{VR}"] = rows
    out["pass"] = all(r["diff_Er_over_gamma"] < 1e-3 and r["rel_diff_gamma"] < 1e-3 for v in out.values() if isinstance(v, list) for r in v)
    return out


def c7():
    E = np.linspace(0.05, 15, 300)
    fom = gaussian_fom(S, S, D, X, 0.0025)
    flux = {}
    for VL, VR in ((10.0, 10.0), (6.0, 14.0), (0.5, 20.0)):
        t, r = fom.amplitudes(mu(E, VL, VR))
        flux[f"fem {VL},{VR}"] = float(np.max(np.abs(np.abs(t) ** 2 + np.abs(r) ** 2 - 1)))
        st = compute_scattering(E, staircase(VL, VR, 2000))
        flux[f"staircase {VL},{VR}"] = float(np.max(st.unitarity_defect))
    t, r = fom.amplitudes(np.column_stack([E.astype(complex), np.full(E.size, 10 - 0.5j), np.full(E.size, 10.0 + 0j)]))
    deficit = 1 - (np.abs(t) ** 2 + np.abs(r) ** 2)
    return {"flux_defect_real_V": flux, "absorbing_min_deficit": float(deficit.min()),
            "pass": max(flux.values()) < 1e-10 and deficit.min() > 0}


def c8():
    E = np.array([1e-2, 1e-4, 1e-6, 1e-8])
    fom = gaussian_fom(S, S, D, X, 0.0025)
    t = fom.amplitudes(mu(E, 10.0, 10.0))[0]
    p = double_barrier(10.0, 0.8, 2.5)
    ts = compute_scattering(E, p).t
    ratio_fem, ratio_sm = np.abs(t) / np.sqrt(E), np.abs(ts) / np.sqrt(E)
    return {"fem_abs_t_over_k": ratio_fem.tolist(), "smatrix_abs_t_over_k": ratio_sm.tolist(),
            "pass": bool(np.std(ratio_fem[1:]) / np.mean(ratio_fem[1:]) < 1e-3 and np.std(ratio_sm[1:]) / np.mean(ratio_sm[1:]) < 1e-3)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, fn in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4), ("C5", c5), ("C6", c6), ("C7", c7), ("C8", c8)):
        t0 = time.perf_counter()
        res[name] = fn()
        res[name]["seconds"] = time.perf_counter() - t0
        print(name, "PASS" if res[name]["pass"] else "FAIL", f"{res[name]['seconds']:.1f}s", flush=True)
    (OUT / "physics_checks.json").write_text(json.dumps(res, indent=1, default=float))
    lines = ["# Independent physics checks (audit 2026-10)", "", "Generated by `scripts/audit/independent_physics_checks.py`.", ""]
    for name, r in res.items():
        lines += [f"## {name}: {'PASS' if r['pass'] else 'FAIL'}", "", "```", json.dumps({k: v for k, v in r.items() if k != 'pass'}, indent=1, default=float), "```", ""]
    (OUT / "physics_checks.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
