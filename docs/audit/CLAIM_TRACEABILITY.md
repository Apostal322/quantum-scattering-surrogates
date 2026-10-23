# Claim traceability (audit, 2026-10-23)

For every headline claim this table gives the document, the experiment and configuration that
produced it, the data file, the analysis script, the commit that added the result, and whether this
audit reproduced it independently. "Reproduced" means recomputed in this audit from code and
configuration. It does not mean re-read from a stored file.

**Phase ownership of files.**

| phase | code | configs | results | docs |
|---|---|---|---|---|
| 1–5 | `qscatter/{physics,surrogates,experiments}/`, `qscatter/{parametric,data,metrics,config}.py`, `scripts/run_exp*.py`, `scripts/analyze_*.py` | `qscatter/config.py`, `results/exp*/config.json` | `results/{physics,exp1,exp1_mlp_no_es,exp2,exp3,exp3b,exp4,exp5,cost,audit}/` | `REPORT.md`, `REPORT_RU.md` §1–10.10, 11–14 |
| 6 (merged, PR #2) | `qscatter/phase6/`, `scripts/phase6/` | `configs/phase6/` | `results/phase6/` | `PHASE6_*`, `REPORT_RU.md` §10.11 |
| 7 (PR #3, open) | `qscatter/phase7/`, `scripts/phase7/` | `configs/phase7/` | `results/phase7/` | `PHASE7_*` |
| audit (this branch) | `qscatter/physics/matching.py`, `scripts/audit/`, `tests/audit/` | `configs/phase7/mlp_diagnostic.json` | `results/audit_2026/`, `results/phase7/mlp_diagnostic/` | `docs/audit/` |

Phase 1–5 result files do not record the git commit they were produced with (Phases 6–7 do: the
`provenance` block of each `results.json`). For Phases 1–5 the commit column gives the commit that
**added the file** (`git log --diff-filter=A`). The 17 headline numbers are recomputed in CI
(`scripts/reproduce_key_results.py`), so their provenance is checked by reproduction.

## Phases 1–5

| # | Claim (where stated) | Experiment / script | Config | Data | Analysis | Result commit | Reproduced in this audit |
|---|---|---|---|---|---|---|---|
| 1 | S-matrix solver: textbook barrier 4e-15, T+R = 1 to 7e-14, ODE ≤ 3e-10 (REPORT §Summary, RU §4.7) | `tests/test_physics.py`, `scripts/make_physics_figures.py` | `qscatter/config.py` | `results/physics/physics_summary.json` | same | 1e7a3c3 | **yes**: independent global matching system agrees to 5e-15 (random multilayers, κa = 50, complex E); unitarity 6.8e-14 recomputed (C1, C7; `results/audit_2026/physics_checks.md`) |
| 2 | Stage B resonances E_n, FWHM, pole widths (REPORT §3 table) | `scripts/make_physics_figures.py`, `physics/poles.py` | Stage B | `results/physics/` | — | 1e7a3c3 | **yes**: 9/9 values in `reproduce_key_results.py` (clean clone and CI) |
| 3 | AAA(t(k)) RMSE 1.8e-7 at N = 20, 3.2e-12 at 40; PCHIP stays ≥ 1e-2 (Exp. 1) | `scripts/run_exp1.py` | `results/exp1/config.json` | `results/exp1/raw_results.jsonl` | `scripts/analyze_exp1.py` | dfbcae3 → f772b9c | **yes** (5 numbers, `reproduce_key_results.py`) |
| 4 | Adaptive segment(E,t) sampling 15/20 vs 20/20 for AAA (Exp. 2–3) | `scripts/run_exp2.py`, `run_exp3.py` | `results/exp2/config.json`, `results/exp3/systems.json` | `results/exp2,3/raw_results.jsonl` | `analyze_exp2/3.py` | c7856f4, f772b9c | no (archived; covered by the 2026-10-13 audit script `scripts/audit.py`, 10/10 PASS) |
| 5 | Detectability map (Exp. 3b) | `run_exp3b_detectability.py` | in script | `results/exp3b/raw_results.jsonl` | `analyze_exp3b.py` | f772b9c | no |
| 6 | Parametric (E, V₀, a, d): MLP 9.4e-3 / 3.5e-3 at 2 048 / 32 768; hybrid 4.5e-5; "resonances found" 66–68/68 (Exp. 4) | `scripts/run_exp4.py` | `qscatter/experiments/parametric_exp.py` | `results/exp4/raw_results.jsonl` | `analyze_exp4.py` | f772b9c → 8bfa81a | **yes, retrained**: MLP 7.8e-3/9.4e-3/1.24e-2 and 2.8e-3/3.5e-3/4.4e-3 per seed, hybrid 4.52e-5, tensor 4.6e-2, identical to the report. **But** the "found" count is lenient; strict count lower (ISSUE P1-EXP4-METRIC; `results/audit_2026/exp4_strict.json`) |
| 7 | Stress test: AAA recovers resonances in 27 unseen systems (Exp. 5) | `scripts/run_exp5.py` | `results/exp5/systems.json` (seed 20261013) | `results/exp5/raw_results.jsonl` | `analyze_exp5.py` | 8bfa81a → 6eb60f8 | no (archived) |
| 8 | FWHM/Γ_p − 1 = 7.8e-5 for isolated resonances, O((Γ/Λ)²) (REPORT §4.7) | Exp. 5 analysis | — | `results/exp5/` | `analyze_exp5.py` | 6eb60f8 | partially: Stage B pole vs FWHM recomputed (row 2) |
| 9 | Cost: ODE + 30 evals + AAA ≈ 670–700× faster than a 20 001-point ODE scan; fast solver beats every surrogate (REPORT §5) | `scripts/benchmark_cost.py` | in script | `results/cost/cost_1d*.json` | — | f0bb248 | **yes in ratio, not in absolute time**: 620× on a different CPU (Xeon 2.8 GHz, load ≈ 1.2); absolute times 1.5–2.2× slower than reported (`results/audit_2026/timing.json`) |

## Phase 6 (merged)

| # | Claim | Experiment | Config | Data | Analysis | Result commit | Reproduced |
|---|---|---|---|---|---|---|---|
| 10 | Phase 6 FEM validated vs S-matrix, O(h²) | `tests/phase6/test_rb.py` | — | — | — | — | **yes** (tests pass in clean clone; the Phase 7 FEM, same scheme, checked against the independent matching system, C2) |
| 11 | 1D: RB greedy reaches the FOM floor with 10 solves; AAA greedy 20 | `scripts/phase6/run.py rb_1d` | `configs/phase6/rb_1d.json` | `results/phase6/rb_1d/results.json` | `scripts/phase6/analyze.py` | 532206e | no (Phase 7 C-1D finds the same ordering: RB 16, adaptive AAA 24, on Gaussians) |
| 12 | 3-parameter (E, V_L, V_R): RB ~2e-9 with 24 solves for any design; RBF/POD ≈ 1 | `run.py rb_3d` | `configs/phase6/rb_3d.json` | `results/phase6/rb_3d/results.json` | same | 81ffb8b | no (RBF/POD "best design" chosen on test data: known Phase 6 limitation, P1-P6-LEAK) |
| 13 | Residual bound ‖A⁻¹‖·‖res‖ holds on 100 % of test points | `run.py rb_3d` | same | same | same | 81ffb8b | **corrected**: ‖A⁻¹‖ from power iteration is a lower estimate (up to 3.4 % low, Phase 7 E; regression test `test_power_iteration_inverse_norm_is_only_a_lower_estimate`) |
| 14 | Basis rank ≈ +1 per resonance, nearly independent of width | `run.py nwidth` | `configs/phase6/nwidth.json` | `results/phase6/nwidth/results.json` | same | de2678f | confirmed and deconfounded by Phase 7 D (fixed geometry, E_max sweep) |
| 15 | Cost: RB break-even ≈ 170 (≈ 4 400 with greedy) queries vs FEM | `scripts/phase6/benchmark_cost.py` | — | `results/phase6/cost/cost.json` | — | 81ffb8b | no |

## Phase 7 (PR #3)

All Phase 7 runs record config, commit and a clean-tree flag in `results/phase7/<exp>/results.json`;
the code they used is unchanged since 64668e7 (`git diff 64668e7 892b96b -- qscatter/phase7/{fom,experiments,protocol,resonances,validation}.py` is empty).

| # | Claim (PHASE7_RESULTS.md) | Experiment | Config | Data | Analysis | Result commit | Reproduced |
|---|---|---|---|---|---|---|---|
| 16 | FEM O(h²) to the ODE (ratio 3.93–4.01), error 4e-5–2e-4 at h = 0.0025, E_r within 0.65 % Γ | `run.py validation` | `validation.json` | `results/phase7/validation/` | `scripts/phase7/analyze.py` | 841a7c2 | **yes, independently**: Richardson FEM agrees with the ODE to 4e-9–2e-8 (C5); FEM poles agree with staircase-S-matrix poles to ≤ 3e-6 Γ in E_r and ≤ 2e-8 in Γ (C6); the FEM error is identified as P1 dispersion, k³h²L/24 (C3, new) |
| 17 | 3-parameter RB 1.2e-9 at 48 solves; RBF/POD/MLP ≈ 1 | `run.py generalization` | `generalization.json` | `results/phase7/generalization/` | same | 892b96b | partially: manifest hashes and disjointness re-verified; MLP failure diagnosed (`results/phase7/mlp_diagnostic/`, ML_EVALUATION_AUDIT.md) |
| 18 | Rectangular → Gaussian transfer reaches FEM-error level at ~2× basis size | same | same | same | same | 892b96b | not rerun; interpretation corrected (no efficiency gain unless rectangular snapshots already exist, P1-B3) |
| 19 | Basis size +1.5–3 per resonance; floor ≈ 5e-11/Γ_min | `run.py basis` | `basis.json` | `results/phase7/basis/` | same | 49ecd79 | not rerun; wording checked (descriptive fit, not a law) |
| 20 | "Certified" bound res/σ_min holds 100 %, overestimates 28–400× | `run.py generalization` | same | same | same | 892b96b | **yes**: holds down to the round-off floor (`test_residual_sigma_min_bound_holds_down_to_the_round_off_floor`); terminology downgraded to "a-posteriori bound evaluated in floating point" (P1-CERT) |
| 21 | 1-D: RB 16, adaptive AAA 24, uniform AAA ~48 queries | `run.py slice1d` | `slice1d.json` | `results/phase7/slice1d/` | same | 841a7c2 | not rerun |
| 22 | Robustness: AAA-uniform noise sensitivity, doublets, coarse data, complex64 | `run.py robustness` | `robustness.json` | `results/phase7/robustness/` | same | f336f22 | not rerun |
| 23 | Cost: RB break-even 2.4e3–1.6e4 (greedy) / 270–570 (Sobol) queries vs FEM | `scripts/phase7/benchmark_cost.py` | `generalization.json` | `results/phase7/cost/` | same | 1999755 | **in ratio**: FEM/RB-single times re-measured 5.9e-4 / 1.7e-4 s (vs 3.6e-4 / 7.9e-5); absolute values machine-dependent |

## Not reproduced in this audit, and why

Experiments 2, 3, 3b and 5, and Phase 6 `rb_1d`, `rb_3d` and `nwidth`, were not rerun. They take
tens of minutes to hours, and their conclusions are either covered by the 17 CI-reproduced numbers
or re-tested in Phase 7 on a new potential family. Their result files are kept unchanged as
historical evidence.
