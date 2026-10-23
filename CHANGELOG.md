# Changelog

All results are kept in the repository history. This file summarises the main stages, including corrections of
earlier, wrong conclusions.

## Audit (2026-10-23, branch `research/phase7-independent-validation`)

Independent scientific, numerical, ML and reproducibility audit; details in `docs/audit/`.

- **No blocking (P0) error found.** Physics re-validated against a new independent matching
  solver, the analytic Born limit, metamorphic relations and an independent pole computation.
- **Corrections of overstated claims (P1):**
  - the Phase 7 "certified" bound is an a-posteriori bound evaluated in floating point;
  - the Phase 6 power-iteration bound was not rigorous;
  - the Experiment 4 "resonances found" metric was lenient (strict count: MLP 61–62/68, spline 26/68);
  - the hybrid is interpolation plus exact composition, not a learned model;
  - rectangular → Gaussian transfer brings no saving unless rectangular snapshots already exist;
  - "3D" meant three parameters.
- **Code fixes (P2):**
  - the ODE reference now fails loudly;
  - surrogate pole search verifies roots;
  - MLP divergence raises explicit errors.
- **Further changes:**
  - bibliography corrected;
  - CI extended to Python 3.10 and 3.13;
  - 15 audit regression tests;
  - MLP failure diagnostic.

## Unreleased: repository publication

- Professional README, figure gallery, experiment, reproducibility and provenance documentation.
- GitHub Actions CI: ruff, the test suite, and smoke tests on Ubuntu and Windows; a scientific reproduction check.
- `python -m qscatter.quickstart`, a one-minute end-to-end demo.
- Distribution renamed to `quantum-scattering-surrogates`; the import name stays `qscatter`.
- Correct minimum dependency versions (NumPy ≥ 2.0, SciPy ≥ 1.15).
- The build artifact `qscatter.egg-info/` is no longer tracked.
- Reference caches are rebuilt automatically in a fresh clone.

## Final scientific review

- Corrected the statement that resonance poles lie on the physical sheet. They lie on the second (unphysical)
  sheet, Im k < 0.
- Clarified the Γ vs FWHM definition, including the Fano caveat. Added the 1D S-matrix and the 1D vs 3D
  distinction.
- Added `scripts/reproduce_key_results.py` (17/17 reproduced) and a second, independent timing run.

## Stress test and parametric surrogates

- Experiment 4 (parametric surrogate) and Experiment 5 (27 unseen systems). Rational-approximant diagnostics.
- **Correction:** the Experiment 4 spurious-peak metric counted near-duplicate grid points as peaks. It was
  recomputed where possible and withdrawn otherwise; a regression test now covers it.
- **Correction:** FWHM read off grid points gave a spurious 2 % deviation from the pole width. With root
  refinement it is 8·10⁻⁵.
- **Correction:** greedy AAA self-certification is reliable only for noise-free data.

## AAA rational approximation and robustness

- Added AAA rational approximation of t(k), exact complex poles, an independent audit (10 checks), and Experiment 3
  (38 systems).
- **Correction:** the earlier claims "the decisive factor is where the samples are, not the model class" and "an
  interpolant cannot recover an unseen peak" are wrong for rational approximation. Amplitude-driven adaptive
  sampling was demoted from the main result to a secondary, convention-dependent effect.

## Initial study

- Validated S-matrix solver, classical baselines, PyTorch MLP, Experiments 1–2.
- **Correction during the work:** the GP baseline collapsed to a degenerate length scale. It is now bounded by the
  sample spacing.
