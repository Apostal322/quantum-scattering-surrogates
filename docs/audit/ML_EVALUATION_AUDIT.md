# ML evaluation audit (audit, 2026-10-23)

**Verdict.**

- No leakage was found in any phase.
- Every learned model is selected on validation data, and inputs are scaled with fixed domain
  bounds, not test statistics.
- The **Phase 7 MLP was under-trained**. With a bounded, validation-only search and longer
  training, its mean test error drops about 3×. That does not change the conclusion: it still
  misses narrow resonances, and it cannot even fit its own training points on the peaks.
- Experiment 4's resonance counts were lenient (P1-EXP4-METRIC).
- The "hybrid" is not a learned model (P1-HYBRID).

Data: `results/phase7/mlp_diagnostic/results.json`, from commit e2074a1 with a clean tree, 3,327 s
on 4 processes. Config, frozen before the run: `configs/phase7/mlp_diagnostic.json`. Code:
`qscatter/phase7/mlp_diagnostic.py`.

## 1. Models in the project and what each learns

| Model (phase) | Input → output | Learned parameters | Information used | Notes |
|---|---|---|---|---|
| MLP 1D (1–5) | E → (Re t, Im t) or T | weights of 1→64→64→2, SiLU, float64 | N values of t | early stopping on interleaved training points, then refit; settings chosen by validation loss (`scripts/mlp_config_selection.py`) |
| MLP 4-param (1–5, Exp. 4) | (E, V₀, a, d) → (Re t, Im t) | 4→128³→2, float32, minibatch Adam | B values of t on Sobol configurations × 32 energies | every 8th configuration validates; targets standardised with training statistics |
| Fourier-feature MLP (4) | same | + scale σ chosen on validation | same | σ = 0.5 always selected; did not help |
| POD + MLP (4) | (V₀, a, d) → POD coefficients | MLP on coefficients | same scans | compression exact, prediction poor |
| GP (1–5) | E or 4-param → t | ARD Matérn hyperparameters by marginal likelihood | ≤ 2,048 points | length-scale lower bound to avoid a degenerate optimum |
| RBF (4, 6, 7) | parameters → t | none (interpolation); kernel chosen on validation in Phase 7 | values of t | — |
| **Hybrid** (4) | (E, V₀, a, d) → t | **none** (tensor-grid cubic interpolation) | single-barrier S-matrices on a grid in (E, V₀, a) **plus the block structure** of the potential | not a neural network; final error ≈ block-interpolation error (4.7e-4 vs 6.9e-4 at B = 2,048; 9.8e-5 vs 7.5e-5; 4.5e-5 vs 1.4e-5); the exact-block ablation reproduces the solver to 8e-15 |
| MLP 3-param (7) | (E, V_L, V_R) → (Re t, Im t) | 3→w×depth→2, SiLU, float64, full-batch Adam, cosine | n values of t on Sobol designs | validation-only selection; this audit's diagnostic below |

## 2. Dataset and training correctness (checked)

| Item | Finding |
|---|---|
| Inputs | scaled to the unit box with **fixed domain bounds** (Phase 7) or fixed bounds `pm.scale` (Exp. 4), not with data statistics |
| Targets | complex t split into (Re t, Im t); Exp. 4 standardises with training mean and std only. Phase 7 targets lie in the unit disk and are not scaled |
| Loss | mean squared error over both components (mean reduction) |
| Early stopping / checkpoint | best validation loss kept: every 250 steps in Exp. 4, every epoch in Phase 7 |
| Seeds | `torch.manual_seed`; deterministic on CPU (new test: two runs with the same seed are bit-identical) |
| Device / precision | CPU only; float32 in Exp. 4, float64 in Phase 7 |
| NaN/Inf | both trainers crashed with opaque errors if no validation loss was finite; they now raise `FloatingPointError` (P2-MLPDIV, tests). Predictions are checked to be finite in the diagnostic. No invalid prediction is silently replaced |
| Serialization | Phase 1–5 MLP save/load tested (`test_mlp_deterministic_and_save_load`); Phase 7 MLPs are not serialized (refit from config) |
| Split overlap | Phase 7: train designs, greedy candidates, validation, test and OOD sets are pairwise disjoint, and the SHA-256 manifest reproduces (new test). Exp. 4: configurations disjoint across splits (`results/exp4/leakage_check.md`) |

## 3. Leakage audit

| Phase | Hyperparameters chosen on | Design or baseline chosen on | Finding |
|---|---|---|---|
| 1–5 | validation (interleaved or held-out configurations) | fixed in advance | no leakage. The reference grid is printed in the MLP-config log but not used for selection |
| 6 | — | **test data** (best RBF/POD design) | known historical limitation (P1-P6-LEAK). Optimistic for the baselines, so it does not favour RB. Superseded by Phase 7 |
| 7 | validation only (RBF kernel, MLP width, depth, learning rate) | pre-declared `baseline_design = sobol` | no leakage. Effectivity points near resonances use test-scan poles for **evaluation only** |
| audit | validation only (MLP grid, RBF kernel) | protocol frozen in config before running | the test set is evaluated once per trained model |

## 4. Phase 7 MLP failure: diagnostic

**Protocol** (frozen before the run):

- Bounded search on the 4,096-point Sobol design: width ∈ {128, 256}, depth ∈ {3, 4},
  lr = 3e-3, 10,000 epochs, patience 1,500. Phase 7 itself used 3,000 epochs and width {64, 128}.
- Selection by validation MSE: **256 × 3** (validation MSE 5.1e-4; 128×4: 5.4e-4; 256×4: 6.3e-4;
  128×3: 1.3e-3).
- The selected model was retrained with 3 seeds at 1,024 and 4,096 points.
- Test errors are split into points within 10 Γ of one of the two narrow resonances of their
  (V_L, V_R) — 47 of 1,024 test points — and all other points.
- Pole positions were interpolated from a 7 × 7 FEM grid; they are used only to classify points.

**Results.** Each cell gives the range over 3 seeds. "Max" is max |Δt|, "mean" is the mean |ΔT|.

| Training data | Test mean | Test max | Near-resonance mean | Far mean | Training max (own points) |
|---|---|---|---|---|---|
| Phase 7 original (128-wide, 3k epochs), 4,096 | 9.1e-3 | 0.76 | — | — | — |
| 1,024 Sobol | 7.2–7.9e-3 | 0.75–0.79 | 6.6–6.8e-2 | 4.3–5.1e-3 | 0.19–0.82 |
| **4,096 Sobol** | **2.6–3.5e-3** | 0.39–0.63 | 2.6–4.7e-2 | 1.5e-3 | 0.40–0.56 |
| 4,096 with half within ±3Γ of a resonance (**oracle**, uses pole positions) | 5.5–8.7e-3 | 0.28–0.44 | 1.5–6.0e-2 | 5.0–6.2e-3 | 0.81 |
| RBF (linear, chosen on validation), same 4,096 Sobol points | 2.2e-2 | 0.78 | 7.4e-2 | 1.9e-2 | — |
| Galerkin RB, 48 FEM solves (Phase 7) | ≈ 1e-10 | 1.2e-9 | — | — | — |

**Diagnosis**, against the causes listed in the brief:

1. **Under-training: yes, partly.** Longer, wider training cut the mean error 3.5× (9.1e-3 →
   2.6e-3) and the worst case from 0.76 to 0.39. The Phase 7 MLP rows understate the MLP and are
   corrected in `PHASE7_RESULTS.md`.
2. **Metric dominated by narrow resonances: yes.** The worst case always lies near a resonance;
   the far-from-resonance error is 17–31× smaller in mean.
3. **Insufficient training coverage: secondary.** Only 173 of 4,096 Sobol points (4 %) lie within
   10 Γ of a narrow resonance. Moving half the budget there (oracle) lowers the worst case by about
   30 % (0.39–0.63 → 0.28–0.44). It improves the near-resonance mean for 2 of 3 seeds only, worsens
   the far error 3–4× and worsens the overall mean.
4. **Expressiveness / optimisation (spectral bias): primary.** Even with 2,133 training points at
   the resonances, the MLP misses **its own training points** on the peaks by up to 0.81. The
   Sobol-trained models miss them by 0.40–0.56. A peak of width Γ ≈ 1e-4–1e-3 on an energy range of
   10 is a feature 1e4–1e5 times narrower than the domain. A smooth MLP trained by Adam does not
   represent it within this budget.
5. **Not caused by:**
   - input scaling (fixed bounds);
   - target representation (t is bounded, |t| ≤ 1);
   - an implementation error (deterministic, finite, validation loss decreasing, training error
     small away from resonances);
   - early stopping (≈ 9,900 epochs run in the search, so the best checkpoint is at the end).
6. **Seed sensitivity:** worst case 0.39–0.63 across seeds at 4,096 points; mean error within 1.4×.
   No failed or diverged run.

**Conclusion.** Among learned models under comparable (t-only) information, the tuned MLP is the
best: about 8× better than the RBF on the same points in mean error, consistent with Experiment 4.
It still does not reconstruct narrow resonances in the three-parameter problem with up to 4,096
points. The comparison with RB is not like for like: RB receives the operator and wave functions,
and 48 solves give 1e-9. This is a valid negative result for this architecture family and budget,
not a statement that neural networks cannot model resonances. A pole-structured output (rational
or Breit–Wigner head) or resonance-aware loss weighting were not tried; they are the natural next
steps.

## 5. Fairness of the comparisons

Methods were compared on four separate axes (`PHASE7_RESEARCH_PLAN.md` §4,
`CLAIM_TRACEABILITY.md`):

- **A. Data efficiency:** the number *and type* of high-fidelity evaluations.
  - RB uses wave functions + operator.
  - RBF, MLP, AAA and PCHIP use t values.
  - The hybrid uses single-barrier S-matrices + structure.
- **B. Prediction quality:** amplitude, |ΔT|, and resonance position, width and recall with the
  strict ±2Γ rule.
- **C. Cost:** offline, online and break-even, measured separately (`REPRODUCIBILITY_AUDIT.md` §4).
- **D. Information requirements:** stated in every comparison table since Phase 7.
- **E. Generalisation:** in-domain, out-of-domain boxes, a new potential family (B3), and new
  heights for the 1D methods.

**Suspiciously large gaps were investigated:**

- RB vs MLP (1e-9 vs 1e-1): explained by information (operator) and by spectral bias (above).
- Hybrid vs MLP (2–3 orders): explained by structural information.
- Exp. 4 recall 67/68: explained by the lenient metric.

The MLP data budget in Phase 7 is larger than RB's (up to 4,096 vs 48 evaluations). That is
deliberate: MLPs are not competitive at RB budgets, and the rows are labelled as such.

## 6. Remaining ML limitations

- Only standard MLP families (SiLU; Fourier features in Exp. 4) were tuned. No pole-structured,
  physics-constrained (unitarity) or resonance-weighted models.
- The tuning grids are small and the budget is bounded: 4 architectures, fixed learning rate,
  10k epochs.
- 3 seeds. Means and ranges are reported, without confidence intervals.
- MLP training times (minutes) are not amortised in the cost tables of Phase 7.
