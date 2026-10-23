# Phase 7 limitations

This file lists what Phase 7 does not show, what it measures less cleanly than its headline
numbers suggest, and one inherited limitation of Phase 6. It is meant to be read together with
`PHASE7_RESULTS.md`.

## 1. Scope of the physics

- **One-dimensional, single-channel, short-range potential.** There are no Coulomb tails, no
  coupled channels and no absorption. Two potential families are covered: a smooth double Gaussian
  (and, in Experiment F, a triple Gaussian), and rectangular barriers in the transfer experiment.
- **Only the heights are parameters.** Width s and separation d enter non-affinely, so they are
  fixed in B/C/E. Experiment D varies them only by building a separate model per value. Making them
  parameters would need empirical interpolation (EIM) or geometric mappings, which were not
  implemented.
- **The reference is a model.** All accuracy numbers below about 1e-4 compare surrogates with the
  FEM at h = 0.0025, not with the exact solution of the Schrödinger equation. The FEM's own error
  is 4e-5 to 2e-4 in t (`PHASE7_NUMERICAL_VALIDATION.md`).
- **Truncation.** The model solves the Gaussian potential truncated to [-X, X] with
  rel_cut = 1e-14. The truncation is measured to change t by less than 3e-13; it is not removed.

## 2. The residual/σ_min error bound (called "certified" in code and data)

- The bound ‖ψ̂ − ψ‖₂ ≤ ‖res‖₂ / σ_min(A(μ)) holds for the **Euclidean norm of the nodal vector**.
  It is not the H¹ or energy norm of the continuous problem, and it bounds the error with respect
  to the FEM solution, not the exact one.
- σ_min is computed **at full order for every parameter**: a banded Hermitian eigenproblem of size
  2N, about 1.3 s for N = 4331. The bound is therefore an *a posteriori certificate*. It is **not**
  an online-efficient certified RB bound in the sense of the RB literature. That would need an
  offline lower bound of the stability constant (e.g. SCM), and none was built.
- The value is "certified" only in floating-point terms. The returned σ is the eigenvalue from
  LAPACK minus a margin of 2m·eps·‖A‖₁, which conservatively covers the eigensolver's backward
  error. No interval arithmetic was used.
- The power-iteration estimate of ‖A⁻¹‖ used in Phase 6 is a *lower* estimate. Phase 7 shows its
  ratio to the certified value but does not use it as a bound.

## 3. Comparison fairness

- **Intrusive vs non-intrusive.** RB uses the assembled FEM operator and wave functions. RBF, the
  MLP, AAA and PCHIP use only values of t. Equal numbers of FOM queries are not equal information.
  The research plan (§4) lists what each method sees.
- **Budgets.** The MLP is trained on its own, larger Sobol designs (48 to 4096 points). At the
  RB-sized budgets (≤ 48) an MLP is not a reasonable model, so these rows are not equal-budget
  rows.
- **Hyperparameters.** RBF kernels and MLP width and learning rate were selected on validation
  data from small grids. Larger searches, other architectures (Fourier features, pole-structured
  networks) and other training schemes could do better; we test one standard MLP, not neural
  surrogates in general.
- **AAA in 1D only.** AAA and PCHIP are compared along fixed-height energy lines. A three-parameter rational
  surrogate (e.g. parametric AAA, or pole matching across heights) was not built.
- **Adaptive AAA rule.** The adaptive AAA design used here (maximum disagreement of consecutive
  fits) is one simple choice among several in the literature. RB greedy uses the residual, which
  is a different kind of information.

## 4. Measurement caveats

- **Pole shift under domain change (A5)** changes X and therefore, slightly, the mesh. The shift
  it reports equals the mesh error and does not isolate the domain effect. The domain effect is
  bounded separately by the ODE truncation study.
- **Resonance matching.** A surrogate pole counts as "matched" if it lies within 2Γ of a FEM pole.
  When an AAA fit has many poles near the axis this test is lenient. `unmatched_poles_near_axis`
  and the pole-position errors should be read together with it.
- **Floating-point floor.** All surrogates stop improving at 3–8e-8 near the narrowest resonance.
  This exceeds the FEM's own round-off (6e-9), and its cause is not fully isolated
  (`PHASE7_NUMERICAL_VALIDATION.md` §5).
- **Timings** are single-thread, median of repeats, on one cloud CPU, measured with other jobs
  idle. They do not transfer to other machines, and the Python overhead of single queries is part
  of what is measured.
- **Seeds.** Noise and jitter studies use 5 seeds and report medians and worst cases, without
  confidence intervals. The three-parameter random-design RB rows use 3 seeds.

## 5. Inherited limitation of Phase 6 (not repeated in Phase 7)

In Phase 6 (`results/phase6/SUMMARY.md`, `docs/PHASE6_NOVELTY_ASSESSMENT.md` §E.2) the design
shown for the RBF and POD-regression baselines was **the best of several designs chosen on the
test data**. That is optimistic for those baselines; it does not favour RB. The Phase 6 artefacts
are left unchanged as a historical record. Phase 7 removes the problem by fixing
`baseline_design = sobol` in the configuration before any evaluation.

## 6. Not done

- No external or experimental data were used; nothing is compared with a published number.
- No coupled-channel, multi-dimensional or time-dependent problem was studied.
- No claim of novelty beyond the coursework level is made (see `PHASE7_NOVELTY_ASSESSMENT.md`).
  The literature check relied on indexed abstracts and metadata; full texts of the closest works
  were not available in this environment.
