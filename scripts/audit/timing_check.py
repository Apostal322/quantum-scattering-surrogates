"""Audit: controlled re-measurement of representative timing claims (Phases 1-7).

    OMP_NUM_THREADS=1 python scripts/audit/timing_check.py

Each quantity is timed `REPEATS` times; min / median / max are reported so that machine-load
variability is visible. Records CPU model, core count, thread settings, numpy/scipy/torch versions
and the BLAS/LAPACK libraries numpy reports. Writes results/audit_2026/timing.json.
Claims checked (source):
  - vectorised S-matrix, Stage B, 20 001 energies: 0.012 s (REPORT.md section 5)
  - ODE solver per energy: 4.4 ms (REPORT.md); 30 ODE calls + AAA vs 20 001-call ODE scan: ~700x
  - Phase 7 FEM single query 3.6e-4 s; RB (n = 24) single 7.9e-5 s, batched 1.7e-5 s per point
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

from qscatter.config import STAGE_B  # noqa: E402
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy  # noqa: E402
from qscatter.phase7.fom import gaussian_fom  # noqa: E402
from qscatter.phase7.protocol import sobol  # noqa: E402
from qscatter.physics.ode_solver import solve_ode  # noqa: E402
from qscatter.physics.smatrix import compute_scattering  # noqa: E402
from qscatter.surrogates.rational import RationalAAA  # noqa: E402

REPEATS = 15


def stats(fn, repeats=REPEATS, per=1):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) / per)
    return {"min": min(ts), "median": statistics.median(ts), "max": max(ts), "repeats": repeats}


def blas_info():
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        return {k: {kk: v.get(kk) for kk in ("name", "version")} for k, v in deps.items() if k in ("blas", "lapack")}
    except Exception as exc:  # noqa: BLE001 - metadata only
        return {"error": repr(exc)}


def main():
    import torch

    cpu = "unknown"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    pot = STAGE_B.potential()
    E = np.linspace(*STAGE_B.E_range, 20001)
    out = {"hardware": {"cpu": cpu, "logical_cores": os.cpu_count(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "torch_threads": torch.get_num_threads(), "python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "torch": torch.__version__, "blas": blas_info(),
                        "load_average_at_start": os.getloadavg()}}
    out["smatrix_scan_20001_s"] = stats(lambda: compute_scattering(E, pot))
    Es = np.linspace(*STAGE_B.E_range, 30)
    out["ode_per_energy_s"] = stats(lambda: [solve_ode(float(e), pot) for e in Es[:5]], repeats=5, per=5)
    t30 = np.array([solve_ode(float(e), pot).t for e in Es])
    out["aaa_fit_and_eval_20001_s"] = stats(lambda: RationalAAA().fit(Es, np.column_stack([t30.real, t30.imag])).predict(E))
    ode_scan = out["ode_per_energy_s"]["median"] * E.size
    pipe = out["ode_per_energy_s"]["median"] * 30 + out["aaa_fit_and_eval_20001_s"]["median"]
    out["ode_scan_projected_s"] = ode_scan
    out["ode_30_plus_aaa_s"] = pipe
    out["speedup_ode_scan_vs_30_plus_aaa"] = ode_scan / pipe
    # Phase 7 FEM and RB
    cfg = json.loads((REPO / "configs" / "phase7" / "generalization.json").read_text())
    fom = gaussian_fom(cfg["s"], cfg["s"], cfg["d"], cfg["X"], cfg["h"])
    lo, hi = np.array(cfg["train_lo"]), np.array(cfg["train_hi"])
    q = sobol(1024, lo, hi, 999)
    out["fem_single_query_s"] = stats(lambda: [fom.amplitudes(m[None]) for m in q[:100]], repeats=5, per=100)
    g = residual_greedy(fom, sobol(cfg["n_candidates"], lo, hi, cfg["seeds"]["candidates"]), 24, 0.5 * (lo + hi))
    rm = ReducedModel(fom, orthonormalize(g.snapshots)[0])

    def single():
        for m in q[:200]:
            c = rm.coefficients(m[None])
            fom.observables(m[None], c @ rm.end0, c @ rm.endL)

    out["rb24_single_query_s"] = stats(single, repeats=7, per=200)
    out["rb24_batched_per_point_s"] = stats(lambda: rm.amplitudes(q, residual=False), repeats=7, per=len(q))
    out["fem_sigma_min_per_point_s"] = stats(lambda: fom.sigma_min(q[0], certified=True), repeats=3)
    out["load_average_at_end"] = os.getloadavg()
    dst = REPO / "results" / "audit_2026"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "timing.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
