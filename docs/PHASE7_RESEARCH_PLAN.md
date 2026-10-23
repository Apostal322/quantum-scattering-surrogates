# Phase 7 research plan: independent validation, generalisation, novelty

> **Audit note (2026-10-23).** This plan is kept as written before the results (pre-registration).
> Two of its terms are corrected elsewhere: "3D" means three *parameters* (E, V_L, V_R) of a
> one-dimensional problem, and the "certified" bound is an a-posteriori bound evaluated in floating
> point, not a computer-verified certificate (`docs/audit/ISSUE_REGISTER.md`, P1-CERT, P2-3D).

**Status:** written on 2026-10-20, after the solver and the protocol code were built and before any
test-set result of Experiments B–F had been looked at. Preliminary convergence checks of the
Experiment A kind were used to choose the mesh (h = 0.0025) and the domain (X = 5.4125). They are
listed in `PHASE7_NUMERICAL_VALIDATION.md`. Everything below is frozen in `configs/phase7/*.json`
and in the code under `qscatter/phase7/`.

## 1. Questions

Phases 1–6 studied rectangular barriers, where an exact S-matrix solver exists. Phase 6 found
that a residual-greedy reduced basis (RB) reproduces t(E, V_L, V_R) with far fewer solver queries
than data-driven interpolation. Phase 7 asks whether that holds up under independent scrutiny:

1. **Physics.** Is a smooth-potential solver with no closed-form reference correct? Are the
   resonances it reports physical rather than mesh, domain or grid artefacts?
2. **Generalisation.** Does the RB advantage survive (B1) a new potential family, (B2) heights
   outside the training box, and (B3) a basis built for a different potential shape?
3. **Fair comparison.** With leakage-free selection and equal access to information, how do FOM,
   RB (Galerkin, LSPG, greedy or a-priori designs), POD regression, RBF, AAA, PCHIP and an MLP
   compare? Queries and computational cost are counted separately.
4. **Basis size.** What sets the number of basis functions: the number of resonances, their widths,
   or the energy window?
5. **Reliability.** Is the reduced residual a *certified* error bound, or only an indicator?
6. **Robustness.** How do noise, coarse solvers, grid jitter, near-degenerate resonances and
   floating-point precision change the picture?
7. **Novelty.** Which findings are already known in the literature?

## 2. Benchmark

V(x) = V_L exp(-(x + d/2)^2 / s^2) + V_R exp(-(x - d/2)^2 / s^2) with s = 0.6 and d = 4 (fixed:
they enter non-affinely). The parameters are mu = (E, V_L, V_R), with ħ²/2m = 1.

- Full-order model (FOM): P1 FEM on a uniform mesh of [-X, X], exact transparent boundary
  conditions for V = 0 outside, and 4-point Gauss quadrature of the potential mass matrices. The
  operator is affine: A(mu) = A0 − E M − k (iB) + V_L M_L + V_R M_R.
- Truncation: X = d/2 + s sqrt(ln(1/rel_cut)). With rel_cut = 1e-14, X ≥ 5.4066; we use X = 5.4125
  so the rectangular-barrier edges of B3 fall on mesh nodes.
- Independent references: the DOP853 ODE integrator on the same domain (real E), and the exact
  S-matrix for rectangular barriers on the same mesh. Poles use Richardson extrapolation in h.

## 3. Protocol (no leakage) — `qscatter/phase7/protocol.py`

- Separate seeds for greedy candidates (2048), validation (256), in-domain test (1024),
  out-of-domain (OOD) test (2 × 256), resonance scans (validation 2, test 6, OOD 4), sampling
  designs and MLP designs. The manifest stores the seeds, sizes and SHA-256 of every set. A test
  checks that the sets are disjoint and reproducible.
- Hyperparameters (RBF kernel, MLP width and learning rate) are chosen **on validation only**, then
  frozen.
- Data-driven baselines are reported on the **pre-declared** design (`baseline_design = sobol`).
  RB is reported on every design, because the design comparison is part of the question. No best
  design is chosen on test data.
- OOD sets (heights 12–14 and 6–8) are always reported separately and labelled as extrapolation.
- **Historical limitation, not repeated:** in Phase 6 the displayed RBF/POD design was selected
  on test data. Phase 6 artefacts are left untouched. The limitation is recorded in
  `PHASE7_LIMITATIONS.md`.

## 4. Experiments

| | Question | Code / config | Main outputs |
|---|---|---|---|
| A | Solver validity, convergence, artefacts | `validation.py`, `validation.json` | ODE tolerance, truncation, mesh convergence of t, r, T, flux, E_r, Γ, cost; energy-grid peak recovery; pole shift under domain change |
| B/C/E | Generalisation, fair comparison, certified bound | `experiments.py`, `generalization.json` | point errors (test, OOD), scan metrics (recall, peak shift / Γ, FWHM, spurious peaks), basis transfer, residual-vs-error statistics |
| C-1D | Energy-only methods (AAA, PCHIP) vs RB on test scans | `slice1d.py`, `slice1d.json` | errors and pole errors at equal query counts; AAA unmatched poles |
| D | Basis size vs complexity | `basis.py`, `basis.json` | resonance count, Γ_min, snapshot rank, greedy n for tolerances, descriptive fit |
| F | Robustness | `robustness.py`, `robustness.json` | noise (5 seeds), coarse data, jitter, doublets, complex64 |
| Cost | Break-even vs FEM, ODE, staircase S-matrix | `scripts/phase7/benchmark_cost.py` | per-stage timings, hardware metadata |

**Information available to each method** (Experiment C):

| Method | Uses |
|---|---|
| FEM | the operator; one sparse solve per query |
| RB Galerkin / LSPG | the affine operator terms (offline and online) plus n snapshots; greedy also uses the residual on candidates |
| POD regression | n snapshots only (no operator online) |
| RBF on t, MLP | n values of t only |
| AAA, PCHIP (1D) | n values of t along one energy line |
| AAA adaptive (1D) | n values of t, the next energy chosen from t data only |

## 5. Hypotheses and how they could fail (stated before the results)

- **H1:** the FEM converges at O(h²) to the ODE solution, and the poles are stable under mesh and
  domain refinement. *Fails if* the errors stall or the poles move by more than Γ.
- **H2:** a greedy RB reaches near the FEM error floor in-domain with tens of snapshots, and beats
  RBF, POD regression and the MLP at equal query counts. *Fails if* any baseline matches it at an
  equal budget on the declared design.
- **H3:** OOD accuracy degrades, more strongly for data-driven methods than for RB. *Fails if* RB
  degrades as much.
- **H4:** a rectangular-barrier basis transfers poorly to Gaussians, but a few Gaussian snapshots
  repair it. *Either outcome is reported.*
- **H5:** the basis size tracks the number of resonances more than their widths. *Fails if* the
  E_max sweep (count changes, widths do not) leaves n flat.
- **H6:** res/σ_min (certified σ_min) always bounds the error, but it is loose near resonances. A
  power-iteration estimate of ‖A⁻¹‖ is only a lower estimate and cannot certify.

## 6. Deliverables

`PHASE7_NUMERICAL_VALIDATION.md`, `PHASE7_RESULTS.md`, `PHASE7_RESULTS_RU.md`,
`PHASE7_NOVELTY_ASSESSMENT.md`, `PHASE7_LIMITATIONS.md`, results under `results/phase7/`, and fast
tests in `tests/phase7/`. The main coursework report (`REPORT_RU.md`) is **not** changed in Phase 7.
