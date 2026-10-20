"""Phase 7 cost benchmark on the smooth double-Gaussian problem (E, V_L, V_R).

    OMP_NUM_THREADS=1 python scripts/phase7/benchmark_cost.py

Times (median of repeats, one thread) every stage of the reduced-basis pipeline and the
reference solvers:
- FEM (P1, production mesh), single and batched queries;
- the DOP853 ODE reference;
- an exact S-matrix solver for a midpoint *staircase* approximation of the Gaussians, with the
  number of steps chosen so that its error against the ODE matches the FEM's own error (the
  staircase is the only way the fast piecewise-constant solver applies to a smooth potential);
- RB offline stages (greedy, snapshots, SVD, assembly) and online queries, AAA/RBF fits.
Break-even query counts Q* = offline / (t_reference - t_RB) use single-query times.

Writes results/phase7/cost/cost.json and cost.md, with hardware/software metadata (CPU model,
core count, library versions; nothing else about the machine).
"""
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
import scipy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qscatter.phase6.experiments import provenance  # noqa: E402
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy  # noqa: E402
from qscatter.phase7.experiments import rbf_fit, rbf_predict  # noqa: E402
from qscatter.phase7.fom import gaussian_double_barrier, gaussian_fom  # noqa: E402
from qscatter.phase7.protocol import sobol  # noqa: E402
from qscatter.phase7.slice1d import _aaa  # noqa: E402
from qscatter.physics.ode_solver import solve_ode_smooth  # noqa: E402
from qscatter.physics.potentials import from_layers  # noqa: E402
from qscatter.physics.smatrix import compute_scattering  # noqa: E402

CFG = json.loads((REPO / "configs" / "phase7" / "generalization.json").read_text())
BASES = [16, 24, 32]
REPEATS = 7


def timed(fn, repeats=REPEATS):
    out = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out)


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def staircase(VL, VR, s, d, X, M):
    edges = np.linspace(-X, X, M + 1)
    mid = 0.5 * (edges[:-1] + edges[1:])
    return from_layers(np.diff(edges), gaussian_double_barrier(VL, VR, s, s, d)(mid), x0=-X)


def main():
    s, d, X, h = CFG["s"], CFG["d"], CFG["X"], CFG["h"]
    lo, hi = np.array(CFG["train_lo"]), np.array(CFG["train_hi"])
    t0 = time.perf_counter()
    fom = gaussian_fom(s, s, d, X, h)
    assembly_fom = time.perf_counter() - t0
    queries = sobol(1024, lo, hi, 999)
    single = queries[:200]

    # --- reference solvers at matched accuracy (heights 10, 10)
    E_chk = np.linspace(0.3, 9.8, 30)
    V = gaussian_double_barrier(10.0, 10.0, s, s, d)
    t_ode = np.array([solve_ode_smooth(e, V, -X, X, rtol=1e-12, atol=1e-15).t for e in E_chk])
    mu_chk = np.column_stack([E_chk, np.full(30, 10.0), np.full(30, 10.0)])
    fem_err = float(np.max(np.abs(fom.amplitudes(mu_chk)[0] - t_ode)))
    stair = []
    for M in (250, 500, 1000, 2000, 4000, 8000, 16000):
        pot = staircase(10.0, 10.0, s, d, X, M)
        err = float(np.max(np.abs(compute_scattering(E_chk, pot).t - t_ode)))
        stair.append({"M": M, "max_abs_dt_vs_ode": err})
    M_eq = next((r["M"] for r in stair if r["max_abs_dt_vs_ode"] <= fem_err), None)
    sm_single = None
    if M_eq is not None:
        def sm_queries():
            for m in single[:50]:
                compute_scattering(np.array([m[0]]), staircase(m[1], m[2], s, d, X, M_eq))
        sm_single = timed(sm_queries, 3) / 50
    ode_single = timed(lambda: [solve_ode_smooth(m[0], gaussian_double_barrier(m[1], m[2], s, s, d), -X, X,
                                                 rtol=1e-10, atol=1e-13) for m in single[:10]], 3) / 10
    fom_single = timed(lambda: [fom.amplitudes(m[None]) for m in single], 3) / len(single)
    fom_batched = timed(lambda: fom.amplitudes(queries), 3) / len(queries)

    # --- offline / online RB
    cand = sobol(CFG["n_candidates"], lo, hi, CFG["seeds"]["candidates"])
    rows = []
    for n in BASES:
        t0 = time.perf_counter()
        g = residual_greedy(fom, cand, n, 0.5 * (lo + hi))
        greedy_total = time.perf_counter() - t0
        snaps = timed(lambda: fom.solve(g.selected), 3)
        svd = timed(lambda: orthonormalize(g.snapshots))
        Vb = orthonormalize(g.snapshots)[0]
        assembly = timed(lambda: ReducedModel(fom, Vb))
        rm = ReducedModel(fom, Vb)

        def rb_single_all():
            for m in single:
                c = rm.coefficients(m[None])
                fom.observables(m[None], c @ rm.end0, c @ rm.endL)

        rb_single = timed(rb_single_all) / len(single)
        rb_batched = timed(lambda: rm.amplitudes(queries, residual=False)) / len(queries)
        rb_batched_res = timed(lambda: rm.amplitudes(queries)) / len(queries)
        offline_sobol = snaps + svd + assembly  # non-adaptive design: n solves + SVD + assembly
        be = {"vs_fem": greedy_total / max(fom_single - rb_single, 1e-30),
              "vs_fem_sobol_design": offline_sobol / max(fom_single - rb_single, 1e-30),
              "vs_ode": greedy_total / max(ode_single - rb_single, 1e-30)}
        if sm_single is not None:
            be["vs_staircase_smatrix"] = (greedy_total / (sm_single - rb_single)) if sm_single > rb_single else None
        rows.append({"n": n, "r": int(Vb.shape[1]), "greedy_total_s": greedy_total,
                     "greedy_fom_s": g.seconds_fom, "greedy_indicator_s": g.seconds_indicator,
                     "snapshots_s": snaps, "svd_s": svd, "assembly_s": assembly,
                     "rb_single_s": rb_single, "rb_batched_s": rb_batched, "rb_batched_with_residual_s": rb_batched_res,
                     "break_even_queries": be})

    # data-driven fits on 32 Sobol points (fit and evaluation cost only)
    Xs = sobol(32, lo, hi, CFG["seeds"]["design_sobol"])
    ts = fom.amplitudes(Xs)[0]
    unit = lambda A: (A - lo) / (hi - lo)  # noqa: E731
    rbf_fit_s = timed(lambda: rbf_fit(unit(Xs), ts[:, None], "thin_plate_spline"))
    rb_ = rbf_fit(unit(Xs), ts[:, None], "thin_plate_spline")
    rbf_eval = timed(lambda: rbf_predict(rb_, unit(queries), 1)) / len(queries)
    E1 = np.linspace(0.1, 10, 32)
    t1 = fom.amplitudes(np.column_stack([E1, np.full(32, 10.0), np.full(32, 10.0)]))[0]
    aaa_fit = timed(lambda: _aaa(E1, t1))
    r1 = _aaa(E1, t1)
    aaa_eval = timed(lambda: r1(np.sqrt(queries[:, 0]))) / len(queries)

    out = {
        "hardware": {"cpu": cpu_model(), "logical_cores": os.cpu_count(), "threads_used": 1,
                     "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "fom": {"nodes": int(fom.nodes.size), "h": h, "X": X, "assembly_s": assembly_fom,
                "single_s": fom_single, "batched_s": fom_batched, "max_abs_dt_vs_ode": fem_err},
        "ode_single_s": ode_single,
        "staircase": {"levels": stair, "M_matched": M_eq, "single_s": sm_single},
        "rb": rows,
        "rbf_32": {"fit_s": rbf_fit_s, "eval_per_point_s": rbf_eval},
        "aaa_32": {"fit_s": aaa_fit, "eval_per_point_s": aaa_eval},
        "provenance": provenance({"generalization_config": CFG, "bases": BASES, "repeats": REPEATS}),
    }
    dst = REPO / "results" / "phase7" / "cost"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "cost.json").write_text(json.dumps(out, indent=1, default=float))
    lines = [f"# Phase 7 cost benchmark\n\nCPU: {out['hardware']['cpu']} (1 thread). FEM N = {fom.nodes.size}.\n",
             "| stage | seconds |", "|---|---|",
             f"| FEM single query | {fom_single:.2e} |", f"| FEM batched, per point | {fom_batched:.2e} |",
             f"| ODE (DOP853, rtol 1e-10) per energy | {ode_single:.2e} |",
             f"| staircase S-matrix (M = {M_eq}) per energy | {sm_single if sm_single is None else f'{sm_single:.2e}'} |"]
    for r in rows:
        lines += [f"| RB n={r['n']}: greedy offline | {r['greedy_total_s']:.2e} |",
                  f"| RB n={r['n']}: single query | {r['rb_single_s']:.2e} |",
                  f"| RB n={r['n']}: batched per point | {r['rb_batched_s']:.2e} |",
                  f"| RB n={r['n']}: break-even vs FEM / ODE / staircase (queries) | "
                  f"{r['break_even_queries']['vs_fem']:.0f} / {r['break_even_queries']['vs_ode']:.0f} / "
                  f"{(lambda v: '–' if v is None else f'{v:.0f}')(r['break_even_queries'].get('vs_staircase_smatrix'))} |"]
    (dst / "cost.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
