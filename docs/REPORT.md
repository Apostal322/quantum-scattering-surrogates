# Surrogate Models for Accelerating Parametric Computations in Quantum Scattering

*Применение суррогатных моделей для ускорения параметрических расчётов в задаче квантового рассеяния*

**Author:** Rostislav Pershin, undergraduate student in Computational Physics. Coursework project; not peer-reviewed.

English research report (portfolio version). The full coursework text in Russian is
[`docs/REPORT_RU.md`](REPORT_RU.md). Every number below is produced by a script in `scripts/`; raw results are in
`results/`; `README.md` lists the commands.

---

## Summary

**Question.** How can the number of expensive exact scattering calculations be reduced while narrow resonances
are preserved? When does machine learning help?

**System.** 1D quantum scattering by piecewise-constant potentials (ħ²/2m = 1). The observables are the complex
transmission amplitude t(E) and T(E) = |t|². There are two test problems: a smooth single barrier, and a
resonant double barrier with resonance widths down to 5·10⁻⁷ in the robustness study. A parametric
extension maps (E, V₀, a, d) → t.

**Reference solver.** Per-layer S-matrices are combined with the Redheffer star product. It is validated against:

| check | agreement |
|---|---|
| textbook barrier formula | 4·10⁻¹⁵ |
| probability conservation (T + R = 1) | 7·10⁻¹⁴ |
| independent DOP853 ODE integration | ≤ 3·10⁻¹⁰ |
| exact complex-plane poles vs. FWHM | consistent, see §3 |

An independent audit with 10 checks passes, including a leakage test with a poisoned reference
(`results/audit/audit_report.md`).

**Main findings.**

1. **Smooth curves.** About 10 exact evaluations suffice. Classical interpolation reaches the
   textbook convergence orders N⁻², N⁻³ and N⁻⁴, and beats a PyTorch MLP by 2–10 orders of magnitude at equal
   budget. The MLP is optimisation-limited at ~10⁻⁴.
2. **Resonant curves: the representation matters more than the sampler or the model class.**
   - Local methods (linear, PCHIP, cubic spline, GP, MLP) on uniform grids all stall at RMSE ≈ 2·10⁻². A
     resonance narrower than the grid spacing is invisible to them.
   - **Rational (AAA) approximation of the complex amplitude as a function of the wavenumber k** works
     because t(k) is meromorphic and resonances are its poles. It recovered every resonance, with RMSE(T) ≤ 10⁻³,
     from 15–30 uniformly spaced evaluations in all 38 systems tested. The narrowest resonance had Γ = 5·10⁻⁷; a
     uniform grid would need ~4·10⁷ points to resolve it.
   - On the original problem, AAA reaches RMSE 1.8·10⁻⁷ with 20 evaluations. Every other method stays above
     10⁻² even with 640.
3. **Self-certification.** A greedy rational sampler with a look-ahead error check certified its own
   convergence in 38/38 systems, with no false certification. At the moment of certification the true error
   was ≤ 2·10⁻¹⁰, or at the noise level.
4. **Adaptive sampling for local interpolants helps, but less and less reliably.** Splitting intervals by the
   change in the complex amplitude beats splitting by T: 15/20 random systems solved vs 10/20. The advantage
   disappears under some phase conventions, so it is an empirical effect, not a guarantee.
5. **Detectability limit.** An unseen resonance can be recovered only if its tail signal Γ/h exceeds the
   larger of the data error and the background-approximation error. The measured detectability map is in §4.4.
6. **Parametric surrogate (E, V₀, a, d) → t.**
   - A plain MLP is the best *generic* method. At equal budget it is ~10× more accurate than tensor-grid
     splines, RBF, GP and POD + MLP (RMSE 3.5·10⁻³ at 32k evaluations). The neural network does become
     competitive in higher dimension.
   - Fourier features and dissipative pretraining did **not** improve it.
   - A structure-exploiting hybrid is 2–3 orders of magnitude more accurate (4.5·10⁻⁵ at 2k evaluations):
     it interpolates (tensor-grid cubic splines, no neural network) only the smooth single-barrier S-matrix and
     composes the double barrier exactly. It is given the block structure of the potential, which no other method
     receives.
   - If each new configuration can be computed on demand, AAA needs only 16 evaluations per configuration
     (6·10⁻¹⁰).
7. **Independent stress test on 27 unseen systems.** These are asymmetric barriers, coupled-well doublets,
   a smooth Gaussian potential, wide intervals, and noisy data.
   - It confirms the AAA result.
   - It locates its limits: under-resolution when there are many poles; instability only in under-resolved
     fits; premature self-certification under 10⁻⁴ noise.
8. **Cost.**
   - With the slow ODE solver (4.4 ms per call), 30 evaluations + AAA reproduce a 20,001-point scan
     **670–700× faster** (two independent measurements).
   - With the fast solver, no surrogate is faster than direct computation.
   - MLP training (~20 s) destroys most of the gain.

### Errata: what changed relative to the first version of this report

| first version claimed | status | evidence |
|---|---|---|
| "On resonant curves the decisive factor is where the solver is evaluated, not the model class"; "an interpolant cannot invent a peak it has not seen" | **Wrong for rational approximation.** It holds only for local methods (splines, GP, MLP). | AAA on t(k) recovers resonances from their analytic tails (§4.2, §4.3). |
| Amplitude-driven adaptive sampling is "the single largest improvement" | **Superseded.** It is a real but secondary, convention-dependent effect. | Exp. 3: 15/20 systems vs 20/20 for AAA; fails with phase reference X = 50 (§4.3). |
| — | The audit found no numerical, leakage or counting errors. The only fix was a pole-extraction detail in the audit itself (spurious Froissart doublets). | `results/audit/audit_report.md` |
| "Self-certification without false alarms" | **Restricted** to noise-free or weakly noisy data. At noise 10⁻⁴ one certification came at 45× the noise level. | Exp. 5 |
| Spurious-peak counts in Exp. 4 (first computation) | **Metric bug**: flat pairs of near-coincident grid points were counted as peaks. Recomputed for the deterministic methods and the MLP; withdrawn otherwise. | `results/exp4/summary.md` |
| FWHM vs pole width deviation of ~2 % (first Exp. 5 analysis) | **Measurement artefact**: grid resolution of 0.02Γ. Root-refined: 7.8·10⁻⁵ for isolated resonances. | §4.7 |

---

## 1. Physics and conventions

The model is −ψ'' + V(x)ψ = Eψ with V = 0 outside [x_L, x_R], and k = √E. The wave is incident from the left.
Amplitudes are defined in **one global coordinate**:

  ψ = e^{ikx} + r e^{−ikx} for x < x_L,  ψ = t e^{ikx} for x > x_R,

with T = |t|², R = |r|², and T + R = 1 for real V. An absorbing layer (Im V < 0) gives T + R < 1. Near an
isolated resonance,

  t ≈ t_bg + c (Γ/2)/(E − E_n + iΓ/2),

i.e. a pole at E_n − iΓ/2. The phase of t rises by ≈ π across each resonance.

**T is not a cross section.** One-dimensional transmission is a dimensionless probability with no angular
dependence. A 3D cross section, σ = (4π/k²) Σ (2ℓ+1) sin² δ_ℓ, has the dimension of an area and needs
partial-wave phase shifts from the radial equation. The extension path is in §7.

## 2. Numerical method (summary)

The layer propagator [[cos qd, sin(qd)/q], [−q sin qd, cos qd]], with q² = E − V, is entire in q². It needs no
branch choice and is regular at the turning point E = V. Each layer is converted to a 2×2 S-matrix, and the
layers are composed with the Redheffer star product. Only bounded quantities are multiplied, so the method is
stable for opaque barriers; it refuses to run only above κd > 600. The method is vectorised at 0.63 µs per
energy.

The same formulas continue analytically to complex E. Exact resonance poles are found by Newton iteration on
1/t(E), starting from AAA candidates. The independent ODE solver (DOP853, rtol 10⁻¹¹) shares no algebra with
the S-matrix code. Full details: `REPORT_RU.md` §4.

## 3. Test problems

* **Stage A:** single barrier, V₀ = 5, a = 1, E ∈ [0.1, 12]. T rises monotonically; the first above-barrier
  resonance is at 14.87.
* **Stage B:** symmetric double barrier, V₀ = 10, a = 0.8, d = 2.5, E ∈ [0.1, 10], chosen by a documented
  parameter scan.

| n | E_n (peak) | Γ_n (FWHM) | Re E_pole | −2 Im E_pole | T_peak (S-matrix / ODE) |
|---|---|---|---|---|---|
| 1 | 0.993817 | 7.471·10⁻³ | 0.993797 | 7.470·10⁻³ | 1 − 3·10⁻¹⁴ / 1 − 3·10⁻¹³ |
| 2 | 3.886801 | 8.742·10⁻² | 3.885902 | 8.729·10⁻² | 1 − 1·10⁻¹⁴ / 1 + 1·10⁻¹³ |
| 3 | 8.384040 | 5.319·10⁻¹ | 8.364136 | 5.185·10⁻¹ | 1 − 2·10⁻¹⁵ / 1 + 6·10⁻¹³ |

The pole widths approach the FWHM in the narrow, Breit–Wigner limit. They deviate by 2.5 % for the broad
resonance, where the background makes the line asymmetric.

![Stage B](../results/physics/stageB_transmission.png)
![Resonances](../results/physics/stageB_resonance_zoom.png)

## 4. Results

### 4.1 Experiment 1: sample efficiency (uniform and random designs; 984 runs)

The full tables, with bootstrap confidence intervals, are in `results/exp1/summary.md` and
`results/exp1/confidence_intervals.md`.

**RMSE of T on the dense reference grid, uniform design:**

| method / target | Stage A N=10 | Stage A N=40 | Stage B N=20 | Stage B N=40 | Stage B N=160 | Stage B N=640 |
|---|---|---|---|---|---|---|
| linear / T | 3.0e-3 | 1.6e-4 | 8.4e-2 | 6.4e-2 | 2.6e-2 | 1.5e-2 |
| PCHIP / T | 2.9e-4 | 3.5e-6 | 8.3e-2 | 6.6e-2 | 2.5e-2 | 1.4e-2 |
| cubic spline / T | 6.2e-5 | 6.5e-8 | 9.0e-2 | 7.5e-2 | 2.5e-2 | 1.4e-2 |
| GP / T | 1.6e-5 | 7.7e-7 | 8.4e-2 | 7.3e-2 | 2.5e-2 | 1.4e-2 |
| MLP / amplitude | 1.4e-3 | 9.8e-4 | 8.3e-2 | 4.5e-2 | 2.8e-2 | — |
| **AAA / amplitude** | **7.0e-9** | **4.8e-16** | **1.8e-7** | **3.2e-12** | **1.5e-14** | **2.4e-15** |
| AAA / T | 7.8e-3 | 1.6e-12 | 1.6e-1 | 8.6e-9 | 8.7e-12 | 3.6e-15 |

![RMSE vs N](../results/exp1/figures/exp1_rmse_uniform.png)

* **Fitting T vs fitting the amplitude depends on the method.**
  - For splines, interpolating T is more accurate, because Re t and Im t oscillate with the propagation phase.
  - For rational approximation, the amplitude is better: |t|² has twice as many poles (mirror pairs), so
    AAA(T) needs about twice as many points.
  - Direct fits of T produce unphysical T̂ < 0 or > 1 with the cubic spline and the GP. Shape-preserving PCHIP
    and linear interpolation never do.
* **The MLP is held back by its optimiser and by spectral bias.**
  - It plateaus near 10⁻⁴ on the smooth curve.
  - An ablation without early stopping (`results/exp1_mlp_no_es/summary.md`) removes the non-monotonic
    artefacts of the early-stopping protocol but not the plateau.
  - On the resonant curve it misses even its own training points on a narrow peak: ΔT up to 0.68 after
    20,000 epochs.

### 4.2 Experiment 2: adaptive sampling on Stage B (544 runs)

PCHIP reconstruction of t; RMSE of T:

| strategy | N=40 | N=80 | N=160 | N=320 | resonances found at N=160 | acquisition time |
|---|---|---|---|---|---|---|
| uniform | 5.2e-2 | 4.5e-2 | 2.8e-2 | 1.9e-2 | 2 | — |
| segment (E, T) | 6.8e-2 | 2.4e-2 | 2.4e-2 | 2.4e-2 | 2 | 0.02 s |
| segment (E, t) | 2.4e-2 | 2.4e-2 | 1.7e-4 | 3.7e-5 | 3 | 0.02 s |
| GP posterior std | 6.6e-2 | 6.8e-2 | 3.2e-2 | 1.8e-2 | 3 | 200 s |
| MLP-ensemble disagreement | 7.7e-2 | 2.5e-2 | 2.4e-2 | — | 2 | 390 s |

**Mechanism.** Across a bracket of half-width δ ≫ Γ around an unseen resonance, the resonant term flips sign:
|Δt| ≈ Γ/δ, while ΔT ≤ 2√T_bg · Γ/δ. Exact numbers at δ = 0.069 around E₁:

| | \|Δt\| | ΔT |
|---|---|---|
| across E₁ | 0.107 | 0.0011 |
| same bracket in the background | 0.0037 | 0.0001 |

A stationary GP gets *worse* as a reconstructor on these multi-scale designs (0.10 → 0.19). It has a single
length scale.

![Designs](../results/exp2/figures/exp2_designs_N160.png)

### 4.3 Experiment 3: robustness across 38 systems (8,360 runs)

The systems vary along five axes:

* a width scan, a = 0.6…2.4, giving Γ_min from 2.5·10⁻² down to 5·10⁻⁷;
* 20 random double barriers;
* 4 energy intervals;
* 3 phase conventions, t·e^{ikX};
* 3 noise levels, 10⁻¹⁰ to 10⁻³.

The reference uses exact poles. Success means: every resonance found **and** RMSE(T) ≤ 10⁻³.

| method | random systems solved (≤ 320 evals) | median N | max N |
|---|---|---|---|
| uniform + PCHIP(t) | 0/20 | — | — |
| uniform + cubic spline(T) | 1/20 | 320 | 320 |
| segment(E,T) + PCHIP(t) | 10/20 | 160 | 320 |
| segment(E,log T) + PCHIP(t) | 3/20 | 240 | 320 |
| segment(E,t) + PCHIP(t) | 15/20 | 80 | 160 |
| uniform + AAA(T) | 20/20 | 20 | 80 |
| **uniform + AAA(t)** | **20/20** | **15** | **30** |
| **AAA greedy + AAA(t)** | **20/20** | **15** | **20** |

* **Width scan.** AAA(t) needs 15–30 evaluations for every Γ_min down to 5·10⁻⁷. Segment(E, t) + PCHIP fails
  once Γ_min ≲ 7·10⁻⁴.
* **Phase convention.** With reference point X = 50, segment(E, t) fails; AAA needs 30 (uniform) or 60
  (greedy) evaluations.
* **Noise 10⁻³.** AAA(t) still finds every resonance from 15 points and reaches the target at 60. AAA(T) fails.
* **Self-certification of the greedy.** It certified in 38/38 systems, always with true error ≤ 2·10⁻¹⁰ or at
  the noise level, and with all resonances found.

![Width scan](../results/exp3/figures/exp3_width_scan.png)
![Random systems](../results/exp3/figures/exp3_random_success.png)

### 4.4 Experiment 3b: when an unseen resonance cannot be detected

AAA(t) on uniform samples, 3 noise realisations per cell:

![Detectability](../results/exp3b/detectability_map.png)

**Without noise, small N limits detection.**
- With N = 20, resonances with Γ < 3·10⁻⁵ are missed. The tail signal Γ/h is below the error with which 20
  points approximate the background.
- That error falls geometrically with N. N = 40 reaches 5·10⁻⁸ and N = 160 reaches 4·10⁻⁹.

**With noise σ, the limit moves roughly in proportion to σ.** At N = 160, resonances are found down to
Γ ≈ 10⁻⁶ for σ = 10⁻⁶ and down to Γ ≈ 2·10⁻³ for σ = 10⁻².

**Rule of thumb:** detectable when Γ/h ≫ max(data error, background-approximation error). The prefactor is not
universal, so we report the measured map rather than a law.

### 4.5 Experiment 4: parametric surrogate (E, V₀, a, d) → t

**Setup.**
- Domain: E ∈ [0.2, 5], V₀ ∈ [5, 10], a ∈ [0.3, 0.7], d ∈ [1.5, 3]; tunnelling regime with 1–2 resonances
  of widths 7·10⁻³…1.
- Training: Sobol configurations × 32 energies. Every 8th configuration validates; the validation data
  count towards the budget.
- Evaluation: 40 held-out test configurations (independent random stream) and 20 out-of-domain ones (one
  parameter 2–10 % beyond the edge), each on pole-resolved reference grids.
- Leakage check: no configuration is shared between any splits. Test configurations sit at about half the
  training spacing from their nearest neighbour (`results/exp4/leakage_check.md`).

Median test RMSE(T), median over 3 seeds:

| method | B = 2,048 | B = 8,192 | B = 32,768 | resonances found (32k; lenient rule, strict counts below) | OOD (32k) |
|---|---|---|---|---|---|
| tensor-grid cubic | 2.0e-1 | 9.8e-2 | 4.6e-2 | 56/68 | 1.2e-1 |
| RBF (thin plate) | 1.7e-1 | 1.2e-1 | 8.0e-2 | 43/68 | 1.5e-1 |
| GP (exact, ≤ 2,048 pts) | 1.6e-1 | 2.9e-1 | — | — | — |
| POD + MLP | 1.5e-1 | 1.1e-1 | 5.0e-2 | 57/68 | 8.9e-2 |
| **MLP** | **9.4e-3** | **4.3e-3** | **3.5e-3** | 67/68 | 7.9e-3 |
| MLP + Fourier features | 1.7e-1 | 4.6e-2 | 9.6e-3 | 61/68 | 1.1e-1 |
| MLP, dissipative pretraining (half the budget lossy) | 3.0e-2 | 1.1e-2 | 9.0e-3 | 62/68 | 1.9e-2 |
| **hybrid: interpolated single-barrier S + exact star product** | **4.5e-5** | **1.3e-5** | **4.7e-6** | **68/68** | **2.3e-5** |
| per-configuration AAA at query time, 16 evaluations per configuration | 6.3e-10, 68/68 (OOD 4.0e-9, 35/35) | | | | |

![Exp4](../results/exp4/figures/exp4_rmse_vs_budget.png)

**Findings.**
- **The MLP outperforms local and kernel interpolators in the four-parameter problem.** Resonance ridges E_n(V₀, a, d) are narrow in
  E and curved in parameter space, which hurts tensor grids and stationary kernels.
- **Fourier features did not help.** Validation always selected the smallest σ, and the error grows more
  strongly with distance to the training set (Spearman ρ = 0.71 vs 0.18 for the plain MLP at B = 2,048).
- **Dissipative pretraining was 2.5–3× worse** than training from scratch at equal total budget, in our 1D
  transfer of the idea.
- **POD compresses well but predicts poorly.** Rank 54–64 represents the scans exactly, but moving
  resonances make the coefficient map hard to learn.
- **The hybrid wins by 2–3 orders of magnitude** and extrapolates well, but it needs the block decomposition.
- **Per-configuration AAA uses a different cost model.** A parametric surrogate with budget B pays off only
  when more than ~B/16 configurations are needed: more than 128 for the MLP at B = 2,048, and then at
  ~10⁷× lower accuracy.
- **Spurious peaks** (corrected metric): 0 for the MLP and the hybrid, 0–10 for the tensor spline.
- **Audit erratum (2026-10).** "Resonances found" counts a resonance when T̂ > 0.5 at one reference point within
  ±Γ, without checking the line shape. With the strict peak-relative rule of Experiments 5–7 (max |T̂ − T| ≤
  0.1 T_peak over ±2Γ), recomputed after retraining with the original seeds (RMSE reproduced exactly): MLP 44–53/68
  at B = 2,048 and 61–62/68 at B = 32,768 (OOD 24–27/35); tensor spline 26/68 (OOD 1/35); hybrid 68/68 (35/35).
  `results/audit_2026/exp4_strict.json`.

### 4.6 Experiment 5: independent stress test (27 unseen systems, equal budgets)

The families were never used for development (seed 20261013):
- 15 asymmetric double barriers;
- 4 coupled-well triple barriers (overlapping doublets);
- 2 smooth double-Gaussian barriers, a 4,000-layer staircase whose O(h²) convergence was verified against
  direct ODE integration of the smooth V(x), with discretisation error 1.6·10⁻⁶;
- 2 wide intervals up to E = 60;
- noise 10⁻⁸ and 10⁻⁴.

Scoring is peak-relative: a resonance counts as reconstructed when max |T̂ − T| within ±2Γ is ≤ 10 % of its
peak. Full tables: `results/exp5/summary.md`.

| family (resonances) | AAA(t), N = 20 / 40 | greedy AAA | segment(E,t) + PCHIP, 160 | cubic(T), 160 | GP(t), 160 | hybrid, 80 / 160 | MLP / FF-MLP, 160 |
|---|---|---|---|---|---|---|---|
| asymmetric (37) | 37/37 / 37/37 | 37/37 | 35/37 | 21/37 | 21/37 | 37/37 / 37/37 | 6/11 / 6/11 |
| doublets (16) | **11/16** / 16/16 | 16/16 | 11/16 | 8/16 | 3/16 | 16/16 / 16/16 | 2/8 / 2/8 |
| smooth (5) | 5/5 / 5/5 | 5/5 | 4/5 | 2/5 | 2/5 | n/a | 1/5 / 2/5 |
| wide intervals (13) | **2/13** / 12/13 (80: 13/13) | 13/13 (39 evals) | 11/13 | 8/13 | 8/13 | 12/13 / 13/13 | 1/4 / 2/4 |
| noise 10⁻⁴ (5) | 4/5 / 5/5 (RMSE 4e-5) | 5/5, **RMSE 2e-3** | 5/5 | 4/5 | 4/5 | — | — |

![Exp5](../results/exp5/figures/exp5_equal_budget.png)

**Rational-approximant diagnostics.** Every approximant pole was classified against the exact Newton-refined
poles (27 systems × 4 budgets × 2 methods):
- With N ≥ 40 all 83 resonance poles were matched (median position error 10⁻¹³Γ), and no resonance was
  missed.
- Froissart doublets (negligible residue or a near-cancelling zero) and non-physical upper-half-plane poles
  occur, but none produced a spurious peak.
- Sensitivity to a 10⁻¹⁰ data perturbation is ≤ 10⁻⁶ in every converged fit. It reached 10⁻⁴…1 only in
  under-resolved fits (window 0.1–60 at N = 40; doublets at N = 20). This makes the perturbation test a
  cheap reliability indicator.

### 4.7 Pole width vs FWHM

Write t = a(E)/(E − E_p) + b(E), with a and b varying on a scale Λ: the distance D to the next pole, or E_r.
First-order terms in (E − E_r)/Λ are odd about E_r. They shift the peak by ~Γ·(Γ/Λ) but change the half-width
only at second order, so FWHM/Γ_p − 1 = O((Γ/Λ)²).

Over the 73 stress-test resonances, with half-maximum crossings root-refined by the exact solver:
- The median deviation is 7.8·10⁻⁵ for isolated resonances (x = max(Γ/D, Γ/E_r) < 10⁻²).
- The deviation grows as x^1.7, with slope ≈ 2 for x < 0.1, as predicted.
- For overlapping doublets (Γ/D ≈ 0.5–2), FWHM/Γ_p = 0.84…2.6. Two poles produce one merged peak, so only
  the pole width (lifetime ħ/Γ_p) remains meaningful.
- A first analysis read the crossings off grid points (resolution 0.02Γ) and reported a spurious 2 %
  deviation. That was corrected.

![Pole vs FWHM](../results/exp5/figures/exp5_pole_vs_fwhm.png)

## 5. Computational cost

Measured on an idle machine, single thread (`results/cost/cost_1d.md`). The target is T(E) on 20,001
energies for Stage B.

| solver | pipeline | calls | solver [s] | acquisition [s] | fit [s] | inference [s] | total [s] | RMSE(T) | speed-up |
|---|---|---|---|---|---|---|---|---|---|
| S-matrix (vectorised) | direct scan | 20,001 | 0.012 | — | — | — | 0.012 | 0 | 1× |
| S-matrix | 30 uniform + AAA | 30 | 0.0004 | — | 0.004 | 0.006 | 0.011 | 2e-11 | 1.1× |
| S-matrix | greedy AAA (24) | 24 | 0.004 | 0.058 | 0.003 | 0.005 | 0.070 | 1e-12 | 0.18× |
| S-matrix | 160 uniform + MLP | 160 | 0.0004 | — | 24.5 | 0.037 | 24.5 | 2.5e-2 | 0.0005× |
| ODE (DOP853) | direct scan | 20,001 | 89.5 | — | — | — | 89.5 | 9e-12 | 1× |
| ODE | 30 uniform + AAA | 30 | 0.117 | — | 0.007 | 0.004 | 0.128 | 5e-10 | **≈700×** (run 2: 670×) |
| ODE | greedy AAA (24) | 24 | 0.098 | 0.069 | 0.003 | 0.003 | 0.174 | 9e-12 | ≈510× |
| ODE | segment(E,t) 160 + PCHIP | 160 | 0.708 | 0.017 | 0.001 | 0.001 | 0.726 | 2e-4 | ≈120× |
| ODE | 160 uniform + MLP | 160 | 0.66 | — | 22.4 | 0.032 | 23.0 | 2.6e-2 | 3.9× |

The ODE solver was first checked against the S-matrix reference: 9·10⁻¹² over the whole grid.

**Break-even** is the number of single-energy queries after which building the surrogate is cheaper than
calling the solver:
- AAA: 28 queries;
- MLP: 5,275 queries with the ODE solver, 1.5·10⁵ with the fast solver.

**Solver-call reduction is not wall-clock acceleration.** It becomes acceleration only when an exact call
costs much more than building the surrogate. Methods with negligible construction cost (AAA, splines, the
hybrid) benefit first.

**Parametric cost** (Experiment 4, fast solver):
- data generation: 0.015–0.85 s;
- training: MLP 105–116 s, Fourier MLP 450–510 s, GP 220–780 s;
- prediction: MLP ≈ 1.5 µs/point, hybrid 2.5–5 µs, RBF and GP 120–270 µs.

With the ODE solver, generating 32,768 calls would take ≈ 140 s. This is projected from the measured cost per
call, not a measured run, and it is comparable to the MLP training time.

## 6. Conclusions

1. **Choose the representation before the model class.** The complex amplitude, treated as a meromorphic
   function of k, combined with rational approximation, reduced the number of exact evaluations for resonant
   observables by 2–3 orders of magnitude compared with every other method, across 65 systems (Experiments 3
   and 5). Its look-ahead self-certification was reliable on noise-free data but premature in one of two
   systems with noise 10⁻⁴.
2. **Adaptive sampling** is useful for local interpolants, especially when it is driven by the amplitude rather
   than by T. Its gain is smaller and depends on conventions and system parameters.
3. **Neural networks** gave no accuracy advantage in 1D. In the four-parameter problem the plain MLP is the best
   *generic* method. Structure-exploiting composition of S-matrices beats it by 2–3 orders of magnitude, and
   so does per-configuration AAA whenever configurations can be computed on demand.
4. **Stress test.** The AAA conclusion holds on 27 unseen systems, including asymmetric barriers, overlapping
   doublets and a smooth potential. Its failure modes are now measured:
   - under-resolution when the window contains many poles;
   - instability only in under-resolved fits;
   - a detectability floor Γ/h ≳ C·max(data error, background error);
   - premature self-certification with noisy data.
5. **Solver-call savings are not wall-clock savings** unless an exact evaluation is expensive (§5).

## 7. Limitations and future work

* **Scope of the potentials.** All potentials are 1D and piecewise constant. Meromorphy of t(k) holds for
  finite-range potentials. Long-range (Coulomb) tails change the analytic structure.
* **The fast solver favours direct computation.** For our fast solver, direct computation is cheaper than any
  surrogate. Wall-clock gains appear only with the slow ODE solver.
* **Literature access.** The texts of the four papers the continuation draws on could not be read in this
  environment because the network policy blocked access. Their methods were implemented from the authors'
  public code and abstracts:
  - adaptive rational approximation (Pradovera, Hiptmair & Perugia 2025);
  - greedy active-learning emulators (Giri et al. 2026);
  - dissipative-relaxation transfer learning (Nam, Park & Jang 2026);
  - POD + NN (Dölz & Henríquez).

  Please verify the citation details against the originals.
* **Noisy data.** The look-ahead tolerance of 10σ certified too early at σ = 10⁻⁴; noisy solvers need a
  different calibration.
* **Parametric extension.** Parametric rational approximation (pole tracking) is not implemented. The
  Fourier-feature σ grid did not go below 0.5. Dissipative pretraining was tried with a single W.
* **Statistics.** 3–5 seeds per configuration. Confidence intervals are reported, and conclusions are drawn
  only where the differences far exceed the spread.
* **Next steps:**
  - a radial solver for phase shifts δ_ℓ(E), with σ(E) built from surrogates of the partial-wave amplitudes
    e^{iδ_ℓ} sin δ_ℓ, which are meromorphic in k and unitary by construction;
  - parametric rational approximation, i.e. tracking the poles E_n(θ) and Γ_n(θ);
  - unitarity-constrained neural surrogates;
  - reduced-basis / POD methods with residual error estimators once the solver returns wavefunctions.
