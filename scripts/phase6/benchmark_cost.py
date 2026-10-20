"""Phase 6 cost benchmark: every stage of the reduced-basis pipeline vs the FEM FOM and the
exact S-matrix solver, on the 3D problem (E, V_L, V_R).

    OMP_NUM_THREADS=1 python scripts/phase6/benchmark_cost.py

Writes results/phase6/cost/cost.json and cost.md. Each timing is the median of `repeats`
repetitions. "Single query" = one parameter point per call (what an interactive user or an outer
optimisation loop pays); "batched" = many points per call (vectorised). Break-even query counts
use single-query times.
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
from scipy.stats import qmc  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qscatter.phase6.fom import FOM, double_barrier_layers  # noqa: E402
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy  # noqa: E402
from qscatter.physics.smatrix import compute_scattering  # noqa: E402

CFG = json.loads((REPO / "configs" / "phase6" / "rb_3d.json").read_text())
N_BASIS = 24  # first budget at which RB reaches the FOM round-off floor (rb_3d results)
REPEATS = 9


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


def main():
    fom = FOM(double_barrier_layers(CFG["barrier_width"], CFG["well_width"]), CFG["h"])
    lo, hi = np.array(CFG["lo"]), np.array(CFG["hi"])
    cand = qmc.scale(qmc.Sobol(3, seed=CFG["seed_candidates"]).random(CFG["n_candidates"]), lo, hi)
    queries = qmc.scale(qmc.Sobol(3, seed=CFG["seed_test"]).random(2048), lo, hi)
    single = queries[:200]

    # --- offline
    t0 = time.perf_counter()
    g = residual_greedy(fom, cand, N_BASIS, 0.5 * (lo + hi))
    greedy_total = time.perf_counter() - t0
    mu_tr = g.selected
    snapshots = timed(lambda: fom.solve(mu_tr), 3)
    S = fom.solve(mu_tr).T
    svd = timed(lambda: orthonormalize(S))
    V = orthonormalize(S)[0]
    assembly = timed(lambda: ReducedModel(fom, V))
    rm = ReducedModel(fom, V)

    # --- online
    fom_single = timed(lambda: [fom.amplitudes(m[None]) for m in single], 3) / len(single)

    def rb_single_all():
        for m in single:
            c = rm.coefficients(m[None])
            fom.observables(m[None], c @ rm.end0, c @ rm.endL)

    rb_single = timed(rb_single_all) / len(single)
    rb_coeff_single = timed(lambda: [rm.coefficients(m[None]) for m in single]) / len(single)
    rb_batched = timed(lambda: rm.amplitudes(queries, residual=False)) / len(queries)
    rb_batched_with_residual = timed(lambda: rm.amplitudes(queries)) / len(queries)
    C = rm.coefficients(queries)
    rb_reconstruct = timed(lambda: fom.observables(queries, C @ rm.end0, C @ rm.endL)) / len(queries)
    rb_residual = timed(lambda: rm.residual_norm(queries, C)) / len(queries)

    pots = [fom.layers.potential({"VL": float(m[1]), "VR": float(m[2])}) for m in single]
    exact_single = timed(lambda: [compute_scattering(np.array([m[0]]), p) for m, p in zip(single, pots)]) / len(single)
    E = np.linspace(0.1, 10, 100000)
    exact_batched = timed(lambda: compute_scattering(E, pots[0])) / E.size

    offline_without_greedy = snapshots + svd + assembly
    offline_with_greedy = greedy_total + svd + assembly  # greedy already includes its FOM solves

    def breakeven(offline, t_ref, t_rb):
        return None if t_rb >= t_ref else int(np.ceil(offline / (t_ref - t_rb)))

    data = {
        "metadata": {
            "cpu": cpu_model(), "machine": platform.machine(), "os": platform.platform(),
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "fom_nodes": int(fom.nodes.size),
            "basis_size": int(V.shape[1]), "n_fom_solves": N_BASIS, "repeats": REPEATS,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "seconds": {
            "fom_single_query": fom_single,
            "exact_smatrix_single_query": exact_single,
            "exact_smatrix_batched_per_energy": exact_batched,
            "snapshots_total": snapshots,
            "svd_basis": svd,
            "reduced_operator_assembly": assembly,
            "greedy_total_including_its_fom_solves": greedy_total,
            "greedy_indicator_sweeps": g.seconds_indicator,
            "offline_total_without_greedy": offline_without_greedy,
            "offline_total_with_greedy": offline_with_greedy,
            "rb_single_query": rb_single,
            "rb_single_query_reduced_solve_only": rb_coeff_single,
            "rb_batched_per_query": rb_batched,
            "rb_batched_per_query_with_residual": rb_batched_with_residual,
            "rb_batched_observable_reconstruction": rb_reconstruct,
            "rb_batched_residual_norm": rb_residual,
        },
        "break_even_queries": {
            "vs_fom_offline_without_greedy": breakeven(offline_without_greedy, fom_single, rb_single),
            "vs_fom_offline_with_greedy": breakeven(offline_with_greedy, fom_single, rb_single),
            "vs_exact_smatrix": breakeven(offline_without_greedy, exact_single, rb_single),
        },
    }
    out = REPO / "results" / "phase6" / "cost"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cost.json").write_text(json.dumps(data, indent=1))
    s, b, m = data["seconds"], data["break_even_queries"], data["metadata"]

    def us(x):
        return f"{x * 1e6:.1f} µs"

    def ms(x):
        return f"{x * 1e3:.1f} ms"

    md = [
        "# Phase 6 cost benchmark (scripts/phase6/benchmark_cost.py)",
        "",
        f"{m['cpu']}; {m['os']}; Python {m['python']}, NumPy {m['numpy']}, SciPy {m['scipy']}; "
        f"OMP_NUM_THREADS={m['OMP_NUM_THREADS']}. FOM {m['fom_nodes']} nodes; RB from {m['n_fom_solves']} "
        f"greedy FOM solves, basis size {m['basis_size']}. Medians of {m['repeats']} repetitions. "
        "On the shared cloud VM used for the reported numbers, repeated runs differed by up to ~1.5x; "
        "compare orders of magnitude, not the last digit.",
        "",
        "| stage | time |", "|---|---|",
        f"| FOM solve, single query | {us(s['fom_single_query'])} |",
        f"| exact S-matrix solver, single query | {us(s['exact_smatrix_single_query'])} |",
        f"| exact S-matrix solver, batched (per energy) | {us(s['exact_smatrix_batched_per_energy'])} |",
        f"| offline: {m['n_fom_solves']} snapshots | {ms(s['snapshots_total'])} |",
        f"| offline: SVD basis | {ms(s['svd_basis'])} |",
        f"| offline: reduced operators + residual QR | {ms(s['reduced_operator_assembly'])} |",
        f"| offline: residual greedy to n = {m['n_fom_solves']} (incl. its FOM solves) | {ms(s['greedy_total_including_its_fom_solves'])} |",
        f"| offline total, without / with greedy | {ms(s['offline_total_without_greedy'])} / {ms(s['offline_total_with_greedy'])} |",
        f"| RB single query (reduced solve + t) | {us(s['rb_single_query'])} |",
        f"| RB single query, reduced solve only | {us(s['rb_single_query_reduced_solve_only'])} |",
        f"| RB batched, per query (without / with residual) | {us(s['rb_batched_per_query'])} / {us(s['rb_batched_per_query_with_residual'])} |",
        f"| RB batched: observable reconstruction / residual norm, per query | {us(s['rb_batched_observable_reconstruction'])} / {us(s['rb_batched_residual_norm'])} |",
        "",
        "| break-even number of queries | value |", "|---|---|",
        f"| vs FOM, offline without greedy | {b['vs_fom_offline_without_greedy']} |",
        f"| vs FOM, offline with greedy | {b['vs_fom_offline_with_greedy']} |",
        f"| vs exact S-matrix solver | {b['vs_exact_smatrix'] if b['vs_exact_smatrix'] is not None else 'never (RB query is slower)'} |",
        "",
    ]
    (out / "cost.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
