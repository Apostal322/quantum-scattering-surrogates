# Scientific novelty review (audit, 2026-10-23)

**Bottom line.** There is **no supported evidence of a globally new scientific result.**

- The methods are published: S-matrix composition, FEM, AAA, reduced-basis Galerkin/LSPG with
  residual greedy, MLPs, Gaussian processes and POD.
- The main qualitative findings are known or expected from theory.

The project's contribution is an independently validated, reproducible, leakage-controlled
comparison on 1D resonant tunnelling. It also contains several measured observations that are
new *within the project* and stated with their scope.

**Literature access.** Bibliographic data for 40 works were verified from search indexes
(`BIBLIOGRAPHY_CHECK.md`). No full text could be opened from this environment, so statements about
what a paper does *not* contain are not made. In particular, whether Maldonado et al. (2025)
emulate energy as a parameter is unverified.

Categories (from the audit brief):

- **A.** Known result.
- **B.** Independent reproduction.
- **C.** New result within this project.
- **D.** Possibly novel, not established as globally new.
- **E.** Unsupported claim requiring correction.

| # | Finding (phase) | Category | Basis / closest prior work | Note |
|---|---|---|---|---|
| 1 | t(E) is meromorphic in k; resonances are second-sheet poles; FWHM ≈ pole width only for isolated resonances, with deviation O((Γ/Λ)²) (1–5) | A | Taylor (1972); Landau–Lifshitz | the measured deviation (7.8e-5 for isolated resonances) is C |
| 2 | AAA on the complex amplitude t(k) recovers every resonance from 15–30 uniform samples, down to Γ = 5e-7 (1–5) | A + B | AAA (Nakatsukasa et al. 2018); rational surrogates for resonant scattering (Pradovera, Hiptmair, Perugia 2025); AAA for resonances (Bruno, Santana, Trefethen 2026) | the systematic 1D benchmark is B/C |
| 3 | The amplitude is a better AAA target than T (\|t\|² has twice the poles) (1–5) | C | follows from the pole count; observed, not proven in general | — |
| 4 | Greedy AAA with look-ahead check certified 38/38 systems, but too early under 1e-4 noise (1–5) | C | the greedy idea is from gMRI (Pradovera et al.) | empirical, no guarantee |
| 5 | Detectability map Γ/h ≳ C·max(data error, background error) (1–5) | C | heuristic; the constant is not universal | — |
| 6 | Local interpolants and small MLPs miss resonances narrower than the sample spacing (1–5, 7) | A | sampling theory; spectral bias (Rahaman et al. 2019) | quantified here (C) |
| 7 | MLP is the best *generic* method in the four-parameter problem (1–5) | C | — | holds for RMSE. The resonance count was **overstated** by a lenient metric (E → corrected: strict 61–62/68) |
| 8 | Hybrid: interpolated single-barrier S-matrix + exact star product beats the MLP by 2–3 orders (1–5) | C | S-matrix composition is A (Redheffer 1962; Ko & Inkson 1988) | uses structural information other methods lack; it is not a neural model (E → wording corrected) |
| 9 | Fourier features and dissipative pretraining did not help (1–5) | C (negative) | Tancik et al. 2020; Nam, Park, Jang 2026 | our 1D transfer of the ideas, small search |
| 10 | Fewer solver calls ≠ less wall time; fast solver beats every surrogate (1–7) | A | standard cost reasoning | measured ratios (C); absolute times machine-dependent |
| 11 | Galerkin RB with residual greedy needs fewer FEM solves than POD regression, RBF or an MLP; it generalises across parameters (6, 7) | A + B | Maldonado et al. 2025; Giri et al. 2026; Hesthaven–Rozza–Stamm 2016 | confirmed on a new potential family |
| 12 | Along one energy line RB (16 queries) ≈ adaptive AAA (24) > uniform AAA (~48) (6, 7) | A | Galerkin RB on an operator polynomial in k is a rational function; MRI/Loewner ↔ moment matching (Pradovera 2020; Schulze et al. 2018) | expected |
| 13 | Basis size grows ≈ 1.5–3 per resonance at fixed geometry, nearly independent of width (6, 7) | C | consistent with meromorphic-structure / n-width arguments; we found no paper with this exact deconfounded sweep, but did not read full texts | a descriptive fit (R² 0.6–0.8 over 19 configurations), not a law |
| 14 | Attainable RB accuracy in double precision ≈ 5e-11/Γ_min (7) | A + C | forward error ≈ eps·κ(A), with σ_min ∝ Γ near a resonance | measured constant (C) |
| 15 | Residual ranks errors well away from resonances (Spearman 0.92–0.98), poorly within ±2Γ (0.37–0.75) (7) | A + C | Sen et al. 2006 (bounds degrade near resonance) | — |
| 16 | res/σ_min bound held on every point, overestimating 28–400× (7) | A + C | textbook bound; SCM literature | the "certified" label was **E → corrected** to an a-posteriori bound evaluated in floating point |
| 17 | Phase 6 "strict" bound with power-iteration ‖A⁻¹‖ (6) | **E → corrected** | power iteration gives a lower estimate | erratum in `REPORT_RU.md` |
| 18 | The P1-FEM error is dominated by dispersion k³h²L/24 (audit) | A | textbook FEM dispersion analysis | relevant to how "error relative to FEM" is read |
| 19 | Rectangular-barrier basis transfers to Gaussians at ~2× basis size (7) | C | ROM adaptation is known (Amsallem & Farhat 2008) | one pair of families; no efficiency gain unless rectangular snapshots already exist |
| 20 | Greedy sampling not uniformly better than Sobol with three parameters (6, 7) | C | — | consistent across Phases 6 and 7 |
| 21 | "Three-dimensional problem (E, V_L, V_R)" (6, 7) | **E → corrected** | three parameters, 1D space | wording |
| 22 | A leakage-free, equal-budget benchmark of RB (Galerkin/LSPG) vs AAA vs PCHIP/RBF vs MLP vs POD regression on smooth resonant tunnelling, with an independently validated FEM (7) | D | no identical study found in indexes | a benchmark, not a discovery. Establishing it as new needs full-text review |

**Claims that must not be made** (unchanged from `PHASE7_NOVELTY_ASSESSMENT.md` §4, re-affirmed):

- that RB or emulators for quantum scattering, greedy residual emulators, AAA pole extraction, or
  the AAA ≈ RB parity are new;
- that the bound is an online-efficient or computer-verified certificate;
- that MLPs are generally inferior;
- that transfer across potential families works in general;
- that RB is a speed-up for small workloads;
- that the work treats three-dimensional or multichannel scattering.

**What would be needed for a publication-level novelty claim:**

1. Full-text reading of Maldonado et al. 2025, Giri et al. 2026, Pradovera 2020 and 2023,
   Nobile & Pradovera 2021, Bruschini (SPARSE) and Sen et al. 2006.
2. A result beyond the benchmark. Two examples: an online-efficient stability lower bound for
   resonant 1D scattering, or parametric pole tracking.
3. A problem class where the reference solver is genuinely expensive: coupled channels or several
   spatial dimensions.
