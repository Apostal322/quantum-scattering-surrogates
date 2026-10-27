"""Audit: re-score Experiment 4 resonances with the strict, peak-relative criterion.

    OMP_NUM_THREADS=1 python scripts/audit/exp4_strict_resonance_metric.py

Experiment 4 counted a resonance as "found" when T_hat > 0.5 at any reference point within +-Gamma of
the pole (`qscatter.parametric.config_metrics`). That says nothing about the line shape. Experiments 5,
6 and 7 use a stricter rule: max |T_hat - T| <= 0.1 T_peak over +-2 Gamma. Experiment 4 stored only
aggregate metrics, not predictions, so this script retrains representative models with the original
code, seeds and budgets (`qscatter.experiments.parametric_exp`), checks that the original RMSE and the
lenient count are reproduced, and adds the strict count. Results: results/audit_2026/exp4_strict.json.
Historical files in results/exp4/ are not modified.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from qscatter import parametric as pm  # noqa: E402
from qscatter.experiments import parametric_exp as px  # noqa: E402

TASKS = [("mlp", 2048, s) for s in (0, 1, 2)] + [("mlp", 32768, s) for s in (0, 1, 2)] + \
        [("hybrid_smatrix", B, 0) for B in px.BUDGETS] + [("tensor_cubic", 32768, 0)]


def strict(ref, T_hat):
    E, T = ref["E"], np.abs(ref["t"]) ** 2
    ok = []
    for r in ref["res"]:
        w = np.abs(E - r["E"]) <= 2 * r["gamma"]
        ok.append(bool(w.any() and np.max(np.abs(T_hat[w] - T[w])) <= 0.1 * T[w].max()))
    return ok


def run(task):
    import torch

    torch.set_num_threads(1)
    method, B, seed = task
    refs = {"test": px._refs("test"), "ood": px._refs("ood")}
    t0 = time.perf_counter()
    if method == "hybrid_smatrix":
        model = pm.HybridSMatrix().fit(*px.HYBRID[B])
        pred = lambda cfg, E_: model.predict(px._xcfg(cfg, E_))  # noqa: E731
    elif method == "tensor_cubic":
        m = px.TENSOR_M[B]
        axes_c = [np.linspace(pm.LO[i], pm.HI[i], m) for i in (1, 2, 3)]
        E = np.linspace(pm.LO[0], pm.HI[0], px.N_E)
        tg = np.empty((m, m, m, px.N_E), complex)
        for i, V0 in enumerate(axes_c[0]):
            for j, a in enumerate(axes_c[1]):
                for k, d in enumerate(axes_c[2]):
                    tg[i, j, k] = pm.CountingSolver(pm.pot(V0, a, d))(E).t
        model = pm.TensorCubic().fit_grid(axes_c + [E], tg)
        pred = lambda cfg, E_: model.predict(px._xcfg(cfg, E_))  # noqa: E731
    else:
        X, t, n_evals, C = pm.scan_dataset(B // px.N_E, px.N_E, seed)
        Xtr, ttr, Xv, tv, _, _ = px.split(X, t, C, px.N_E)
        model = pm.MLPParam(seed=seed).fit(Xtr, ttr, Xv, tv)
        pred = lambda cfg, E_: model.predict(px._xcfg(cfg, E_))  # noqa: E731
    fit_s = time.perf_counter() - t0
    out = {"method": method, "budget": B, "seed": seed, "fit_seconds": fit_s}
    for kind, rs in refs.items():
        rmse, lenient, strict_ok, n_res = [], 0, 0, 0
        for ref in rs:
            T_hat = np.abs(pred(ref["cfg"], ref["E"])) ** 2
            m = pm.config_metrics(ref, T_hat)
            rmse.append(m["rmse_T"])
            lenient += m["n_det"]
            s = strict(ref, T_hat)
            strict_ok += sum(s)
            n_res += len(s)
        out[kind] = {"median_rmse_T": float(np.median(rmse)), "lenient_found": lenient, "strict_resolved": strict_ok, "n_res": n_res}
    print(method, B, seed, json.dumps({k: out[k] for k in ("test", "ood")}), flush=True)
    return out


if __name__ == "__main__":
    with Pool(2) as pool:
        rows = pool.map(run, TASKS)
    dst = REPO / "results" / "audit_2026"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "exp4_strict.json").write_text(json.dumps(rows, indent=1))
