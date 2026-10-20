# Reproducibility checklist

| item | status | where |
|---|---|---|
| Exact library versions recorded | yes | `requirements-lock.txt` (Python 3.13, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.1, scikit-learn 1.9.1) |
| Minimum versions declared | yes | `pyproject.toml` (NumPy ≥ 2.0 for `np.trapezoid`, SciPy ≥ 1.15 for `scipy.interpolate.AAA`) |
| Fixed random seeds | yes | experiment modules; summarised in [`EXPERIMENTS.md`](EXPERIMENTS.md) |
| Deterministic PyTorch | yes | `torch.use_deterministic_algorithms(True)`, CPU only |
| Numerical precision documented | yes | physics float64; 1D MLP float64; parametric MLP float32 |
| Solver calls counted, not assumed | yes | `CountingSolver`; audit check 8 |
| No reference-data leakage | yes, tested | audit checks 5/6 (poisoned reference); `results/exp4/leakage_check.md` |
| Raw results stored | yes | `results/exp*/raw_results.jsonl` |
| Figures regenerable from raw results | yes | `scripts/analyze_*.py`, `scripts/make_physics_figures.py` |
| Headline numbers re-checked automatically | yes, 17/17 | `scripts/reproduce_key_results.py` → `results/reproduction_check.md`; runs in CI |
| Independent audit | yes, 10/10 | `scripts/audit.py` → `results/audit/audit_report.md` |
| Checkpoint provenance | yes | `meta` in every `results/exp1/models/*.pt` |
| Bit-for-bit reproduction | verified for adaptive designs and an Experiment 1 MLP run | audit checks 10a/10b |
| Wall-clock timings | reproducible in order of magnitude only (±30–50 % between runs) | `results/cost/cost_1d.md`, `cost_1d_run2.md` |
| Known corrected errors documented | yes | `docs/REPORT.md` (errata), `docs/REPORT_RU.md` §12.2, [`CHANGELOG.md`](../CHANGELOG.md) |

## Phase 6 (reduced basis)

| item | status | where |
|---|---|---|
| Configurations | yes | `configs/phase6/*.json` (seeds for candidates, test, scans, designs are distinct) |
| Provenance in every result | yes | `results/phase6/*/results.json` → `provenance` (config, git commit, `code_modified_since_commit`, versions) |
| No test data in selection | yes, tested | greedy uses only the residual on candidates; `tests/phase6/test_rb.py`, `test_experiments.py` |
| End-to-end smoke test of the runners | yes | `tests/phase6/test_experiments.py::test_experiment_runners_complete_end_to_end_on_a_small_config` |
| Figures regenerable | yes | `python scripts/phase6/analyze.py` → `results/phase6/figures/`, `SUMMARY.md` |
| Timings | order of magnitude (up to ~1.5× between runs on a shared VM) | `python scripts/phase6/benchmark_cost.py` → `results/phase6/cost/` |

```bash
OMP_NUM_THREADS=1 python scripts/phase6/run.py nwidth    # ~5 min
OMP_NUM_THREADS=1 python scripts/phase6/run.py rb_1d     # ~3 min
OMP_NUM_THREADS=1 python scripts/phase6/run.py rb_3d     # ~15 min
OMP_NUM_THREADS=1 python scripts/phase6/benchmark_cost.py
python scripts/phase6/analyze.py
```
