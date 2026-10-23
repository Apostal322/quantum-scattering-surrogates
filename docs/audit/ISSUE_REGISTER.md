# Issue register (audit, 2026-10-23)

Severity:

- **P0:** blocking scientific error.
- **P1:** major methodological or software defect, or a misleading interpretation.
- **P2:** reliability or documentation problem.
- **P3:** optional improvement.

Every resolved item has a traceable change (commit on `research/phase7-independent-validation`)
and a test or an independent numerical check.

## P0 — blocking scientific errors

**None found.** The audit searched for each P0 class in the brief:

- incorrect equations or boundary conditions;
- invalid references;
- test-data leakage in confirmatory experiments;
- false physical claims;
- incorrect resonance calculations;
- fabricated or inconsistent results.

None was found. Evidence: `PHYSICS_VALIDATION.md` (C1–C8 pass), the leakage checks below, the
17/17 headline numbers reproduced in a clean clone, and the exact reproduction of Experiment 4
aggregates. One P0 candidate, an unjustified "certified" bound, was examined and downgraded to
P1-CERT: the mathematics is valid and only the label overstated it.

## P1 — major

| ID | Location | Reproduction | Actual | Expected | Impact | Correction | Verification | Status |
|---|---|---|---|---|---|---|---|---|
| P1-CERT | `phase7/fom.sigma_min`, `PHASE7_RESULTS*.md`, `analyze.py`, figure E | read the docstring and docs; compare σ_min with dense SVD and time it | "certified bound" / "certified σ_min"; cost not stated | the bound is exact in exact arithmetic but evaluated in floating point with a heuristic eigensolver margin and no rounding analysis of the residual; each evaluation ≈ 3,800 FEM solves | readers could take it for a rigorous, cheap online certificate | docstring, docs, figure label and summary reworded; cost stated; data keys kept for compatibility | σ_min vs SVD ≤ 1.4e-8 rel.; bound holds to the round-off floor with margin ≥ 22× (`tests/audit`, 2 tests); timing in `results/audit_2026/timing.json` | **resolved** |
| P1-BOUND-P6 | `REPORT_RU.md` §10.11; `PHASE6_NOVELTY_ASSESSMENT.md`; `results/phase6/SUMMARY.md` | `phase6/baselines.inverse_norm` vs 1/σ_min | "строгая оценка ‖A⁻¹‖·‖res‖ holds on 100 %" with ‖A⁻¹‖ from power iteration | power iteration is a lower estimate (ratio 0.966–0.999998); the inequality held empirically but was not a bound | overstated rigour in the merged report | "строгая" removed and erratum row added in `REPORT_RU.md` §12.2; Phase 6 artefacts left unchanged as history | `test_power_iteration_inverse_norm_is_only_a_lower_estimate` | **resolved** (docs) |
| P1-EXP4-METRIC | `qscatter/parametric.config_metrics` → `REPORT*.md` §10.6 / §4.5, README | `python scripts/audit/exp4_strict_resonance_metric.py` | "resonances found" = T̂ > 0.5 at one point within ±Γ; MLP 66–68/68, tensor spline 56/68 | the peak-relative rule of Exp. 5–7 (max \|ΔT\| ≤ 0.1 T_peak over ±2Γ): MLP 44–53/68 (B = 2,048), 61–62/68 (32,768; OOD 24–27/35); tensor 26/68 (OOD 1/35); hybrid 68/68 | recovery of resonances by MLP and especially by splines overstated; method ranking unchanged | errata in `REPORT_RU.md` §10.6 and §12.2, `REPORT.md`, README; historical `results/exp4/` unchanged; new data in `results/audit_2026/exp4_strict.json` | retraining reproduces every original median RMSE and lenient count exactly | **resolved** (erratum) |
| P1-HYBRID | README §summary, §skills; `REPORT*.md` summaries | read `qscatter/parametric.HybridSMatrix` | "physics-informed hybrid", "learned single-barrier S-matrix" | tensor-grid **cubic interpolation** (no neural network) of the smooth single-barrier S-matrix in (E, V₀, a), plus exact composition; it receives the block structure, which no other method does | readers could credit a neural model with an advantage that comes from structural information | wording corrected in README, `REPORT.md`, `REPORT_RU.md`; information asymmetry stated | ablation: final error ≈ block-interpolation error (4.7e-4 vs 6.9e-4 at B = 2,048; 4.5e-5 vs 1.4e-5 at 32,768); exact blocks reproduce the solver to 8e-15 (`test_hybrid_smatrix_composition_is_exact`) | **resolved** |
| P1-B3 | `PHASE7_RESULTS*.md` item 5 | compare transfer rows with native greedy rows | "transfer works better than expected" | a rectangular snapshot costs the same FEM solve; native 32 solves give 3.9e-6 vs rectangular 48 → 1.2e-4. Transfer helps only if a rectangular snapshot set already exists | possible misreading as an efficiency gain | efficiency verdict added | numbers from `results/phase7/generalization/` | **resolved** |
| P1-MLP7 | `PHASE7_RESULTS*.md`: MLP max \|Δt\| ≈ 0.77 up to 4,096 points | `python scripts/phase7/run.py mlp_diagnostic` | strong negative conclusion from one small grid, 3000 epochs, one seed | bounded validation-only diagnostic: architecture, training length, seeds, near/far error split, train error, equal-data RBF, oracle data | the conclusion might reflect under-training, not a limit of information | diagnostic run (see `ML_EVALUATION_AUDIT.md`) | `results/phase7/mlp_diagnostic/results.json` | see `ML_EVALUATION_AUDIT.md` |
| P1-P6-LEAK | Phase 6 `rb_3d`: RBF/POD design chosen on test data | — | historical, already documented in Phase 7 | baseline design fixed before evaluation | optimistic for the baselines (does not favour RB) | Phase 7 protocol; erratum already present | Phase 7 manifest verified: SHA-256 matches the split code; all designs disjoint from validation and test (`test_committed_phase7_manifest_matches_the_split_code_and_sets_are_disjoint`) | **documented, superseded by Phase 7** |

## P2 — reliability and documentation

| ID | Location | Actual | Correction | Verification | Status |
|---|---|---|---|---|---|
| P2-ODE | `physics/ode_solver.solve_ode_smooth` | ignored `sol.success`; accepted E ≤ 0 | raises `RuntimeError` / `ValueError` | `test_ode_reference_refuses_invalid_input_instead_of_returning_garbage` | resolved |
| P2-POLE | `phase7/slice1d.secant_pole` | returned the best iterate even without a root | same acceptance test as `find_pole`; callers drop `None` | `test_secant_pole_reports_none_instead_of_a_stalled_iterate`; archived RB pole errors (≤ 2e-8 Γ) confirm the archived poles were roots | resolved |
| P2-MLPDIV | `phase7/mlp.fit_mlp` | on NaN loss, `load_state_dict(None)` raised an opaque TypeError | explicit `FloatingPointError`; optional depth and history (default behaviour bit-identical) | `test_mlp_training_is_deterministic_finite_and_fails_loudly_on_divergence` | resolved |
| P2-3D | `REPORT_RU.md` (2×), Phase 7 docs, `phase7/analyze.py`; `results/phase6/SUMMARY.md` | "трёхмерная задача" / "3D" for three parameters of a 1D problem | reworded to "three-parameter"; generated Phase 6 summary left unchanged (historical) | text search | resolved (docs) |
| P2-FEM-DISP | Phase 7 docs | FEM error ascribed to "discretisation" without mechanism | documented as P1 dispersion, k³h²L/24; consequences for larger domains and energies stated | `test_fem_free_space_error_is_the_p1_dispersion_phase_error` | resolved |
| P2-BIB | `REPORT_RU.md` §14 | Griffiths edition/co-author mixed; Lei title and status; Dölz & Henríquez, Bruno et al. unpublished versions; missing years and DOIs | corrected or completed (`BIBLIOGRAPHY_CHECK.md`) | search-index verification (no full texts) | resolved |
| P2-README | README | "75 tests", "57 unit…" | 98 tests (after audit) | `pytest` | resolved |
| P2-PROV15 | `results/exp1…exp5`, `results/cost` | result files carry no commit hash | traceability by the commit that added each file; 17 headline numbers re-checked in CI | `CLAIM_TRACEABILITY.md` | documented (not retro-fixable) |
| P2-TIMING | REPORT §5, Phase 6/7 cost | absolute times from one VM | re-measured on another VM: absolute times 1.5–2.2× slower, ratios stable (620× vs 670–700×) | `results/audit_2026/timing.json` | documented |
| P2-PYVER | `pyproject.toml` (requires-python ≥ 3.10) vs CI (3.11, 3.12) | declared minimum untested | CI matrix extended with Ubuntu 3.10 and 3.13 | CI run on this branch | resolved |
| P2-TORCH-INDEX | install instructions | the PyTorch CPU index is unreachable from some networks (it was refused by the audit proxy) | README already documents the PyPI fallback; the clean-room install used it (torch 2.14.1) | clean-room log | documented |

## P3 — optional

| ID | Note | Status |
|---|---|---|
| P3-MLPSEL | `scripts/mlp_config_selection.py`: "tied on Stage B" is generous (validation loss of the chosen config 1.5× the best at N = 80) | noted |
| P3-ADAM | REPORT_RU §10.10 attributed the MLP plateau to Adam without a test | softened to a hypothesis |
| P3-HYB-BUDGET | the hybrid "B = 8,192" row used 9,216 evaluations (+12.5 %) | noted (stated in `results/exp4/summary.md`) |
| P3-SOBOL | Sobol designs of non-power-of-2 size raise a balance warning | cosmetic; sizes are fixed in configs |
| P3-FEM-BATCH | Phase 7 FEM "batched" queries loop in Python and are not faster than single queries | noted in cost docs |
