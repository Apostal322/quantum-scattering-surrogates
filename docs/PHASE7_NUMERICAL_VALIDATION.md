# Phase 7 numerical validation (Experiment A)

**Verdict.** The Phase 7 FEM for the smooth double-Gaussian barrier is validated against two
references that share no discretisation with it:

- Against the ODE, its error falls as O(h²) (ratio 3.93–4.01 per halving, over six meshes).
- It conserves flux to ≤ 1e-11.
- On rectangular barriers it converges to the exact S-matrix at second order on the same mesh.

At the production mesh (h = 0.0025, about 4330 nodes), the error is:

- **transmission amplitude:** |Δt| ≤ 2.1e-4;
- **resonance energy:** |ΔE_r| ≤ 6.5e-3 Γ;
- **pole width:** relative error ≤ 2e-5.

Every resonance in the window is a stable pole of t: it persists under mesh refinement, under a
larger domain, and on fine energy grids. No artefact poles or peaks were found. These numbers set
the **physical error floor** of every surrogate in Phase 7: agreement with the FEM below about
1e-4 is agreement with a model, not with the physics.

Source: `results/phase7/validation/results.json`, produced by `python scripts/phase7/run.py validation`.

- Clean tree at commit 1965294. The FEM, pole and ODE code it used is unchanged since 64668e7.
- Config: `configs/phase7/validation.json`.
- Tables: `results/phase7/SUMMARY.md` §A. Figure: `results/phase7/figures/A_mesh_convergence.png`.

## 1. References and what each one checks

| Reference | Independent of the FEM in | Checks |
|---|---|---|
| DOP853 ODE integration (`qscatter.physics.ode_solver.solve_ode_smooth`), backward from the transmitted wave, rtol 1e-12 | discretisation (adaptive Runge–Kutta, no mesh), boundary treatment | t, r, T, flux on the real axis, same truncated domain |
| Exact S-matrix for rectangular barriers (`qscatter.physics.smatrix`) | everything (closed form per layer) | the FEM code path for indicator-function potentials on the same mesh (test `test_fom_matches_exact_smatrix_for_rectangular_barriers_on_the_shared_mesh`) |
| Richardson extrapolation of FEM poles from h = 0.00125 and 0.000625 | not independent (same method), but removes the leading O(h²) error | complex poles E_r − iΓ/2, where the ODE solver cannot reach (complex E) |

The ODE reference is itself accurate. Lowering rtol from 1e-12 to 1e-11 changes t by ≤ 6e-11, and
its flux defect is ≤ 5e-13. Both are six orders of magnitude below the FEM error at the production
mesh.

## 2. Domain truncation

A Gaussian never vanishes, so the model solves V on [-X, X] with exact transparent conditions for
V = 0 outside. We used X = d/2 + s sqrt(ln(1/rel_cut)) and measured the effect with the ODE, so the
mesh plays no role. The table gives max |Δt| relative to rel_cut = 1e-16, over 60 energies in
[0.15, 10]:

| rel_cut | 1e-4 | 1e-6 | 1e-8 | 1e-10 | 1e-12 | 1e-14 |
|---|---|---|---|---|---|---|
| V = (10, 10) | 3.8e-5 | 3.5e-7 | 2.2e-9 | 2.3e-11 | 6.1e-13 | 2.7e-13 |
| V = (14, 14) | 4.5e-5 | 1.7e-7 | 1.6e-9 | 2.9e-11 | 3.3e-13 | 1.5e-13 |
| V = (8, 14) | 2.0e-5 | 2.0e-7 | 1.1e-9 | 8.7e-12 | 1.7e-13 | 1.1e-13 |

The truncation error is proportional to rel_cut and reaches the round-off level near 1e-12. Phase 7
uses rel_cut = 1e-14, so truncation contributes ≤ 3e-13: negligible.

## 3. Mesh convergence against the ODE (same X)

The table gives max over 60 energies, and the Richardson-referenced pole errors of all three
poles in [0.1, 10].

| h | nodes | V = (10, 10): abs Δt | V = (14, 14): abs Δt | V = (8, 14): abs Δt | flux defect (max) | max abs ΔE_r / Γ (any config) | max rel. err Γ |
|---|---|---|---|---|---|---|---|
| 0.02 | 542 | 9.1e-3 | 1.3e-2 | 2.7e-3 | 3.6e-14 | 4.1e-1 | 1.2e-3 |
| 0.01 | 1083 | 2.3e-3 | 3.3e-3 | 6.8e-4 | 1.3e-13 | 1.0e-1 | 3.0e-4 |
| 0.005 | 2164 | 5.7e-4 | 8.2e-4 | 1.7e-4 | 5.7e-13 | 2.6e-2 | 7.5e-5 |
| **0.0025** | **4327** | **1.4e-4** | **2.1e-4** | **4.3e-5** | 7.3e-13 | **6.5e-3** | **1.9e-5** |
| 0.00125 | 8652 | 3.6e-5 | 5.1e-5 | 1.1e-5 | 3.4e-12 | 1.6e-3 | 4.7e-6 |
| 0.000625 | 17303 | 8.9e-6 | 1.3e-5 | 2.7e-6 | 1.1e-11 | 4.0e-4 | 1.2e-6 |

- The error ratio per halving is between 3.93 (coarsest pair) and 4.01 for t, r and T in every configuration, approaching 4.00 on fine meshes: clean O(h²).
  Nothing stalls before h = 0.000625.
- The flux defect stays at the round-off level (it grows like N·eps). The discrete scheme
  conserves |t|² + |r|² = 1 by construction, so flux is a check on round-off, not on accuracy.
- The worst relative position error is for the narrowest resonance, (14, 14) at E_r ≈ 1.0616 with
  Γ = 1.3e-4. At h = 0.0025 its peak is displaced by 0.0065 Γ.
- The solve time grows linearly with N: 3–5e-4 s per energy at h = 0.0025, for a tridiagonal
  complex solve plus assembly.

The production mesh is **h = 0.0025**. It is the coarsest mesh at which the resonance energies are
correct to better than 1% of Γ for the narrowest resonance in the parameter box. The generalisation
experiments use X = 5.4125 instead of 5.4066, so that rectangular-barrier edges fall on nodes (B3).
That gives 4331 nodes; the truncation table shows the change does not matter.

## 4. Physical resonances or artefacts?

| Check | Result |
|---|---|
| Pole count under refinement (6 meshes, 3 configs) | 3 poles in [0.1, 10] at every h; no pole appears or disappears |
| Pole movement under refinement | O(h²), extrapolating to fixed values (below) |
| Pole movement when the domain grows (rel_cut 1e-14 → 1e-16, at h ≈ 0.0025) | ≤ 2e-5 absolute. **Caveat:** X changes, so the mesh changes slightly; the shift equals the mesh error of §3 and is dominated by it. The pure domain effect is bounded by §2 (≤ 3e-13 on the real axis) |
| Peaks of T on uniform energy grids (find_peaks, prominence 0.05) | 1000 points (spacing 1e-2) miss the narrowest resonance in 2 of 3 configs; 20 000 points find all of them; **no grid finds a peak without a pole** |
| Flux | ≤ 1.1e-11 on all meshes |

Reference poles (Richardson from the two finest meshes), E_r − iΓ/2:

| V_L, V_R | resonance 1 | resonance 2 | resonance 3 |
|---|---|---|---|
| 10, 10 | 0.958456 − 3.46e-4 i | 3.519137 − 8.72e-3 i | 6.953409 − 1.107e-1 i |
| 14, 14 | 1.061625 − 6.68e-5 i | 3.925056 − 1.885e-3 i | 7.875473 − 3.090e-2 i |
| 8, 14 | 0.971518 − 5.75e-4 i | 3.561451 − 1.498e-2 i | 7.040276 − 1.773e-1 i |

The Γ_p = −2 Im E_pole values are pole widths. For an isolated, symmetric resonance they approach
the FWHM of the T peak, but they differ for broad or asymmetric ones (e.g. resonance 3 of (8, 14),
whose peak T is well below 1). Scan metrics in the results always report both.

**The danger is under-resolution, not invention.** A uniform energy grid that is too coarse misses
narrow resonances: Γ = 1.3e-4 against a spacing of 1e-2. A resonance-resolved scan therefore needs
the pole positions first. All Phase 7 test scans add ±20 Γ windows (401 points each) around every
FEM pole.

## 5. Floating-point floor of the FEM itself

Near the narrowest resonance of a test scan (Γ = 4.3e-4), two backward-stable solvers of the same
FEM system disagree by 6e-9 in t. The solvers were LAPACK banded LU and SuperLU. This is the FEM's
round-off level, about eps·cond(A) with cond(A) ≈ 4e7 at the resonance.

Every surrogate in Phase 7 reaches a separate floor of 3–8e-8 (depending on the scan), at the same narrowest-resonance
peak. This holds for greedy RB with Galerkin or LSPG, with 19 or 26 basis vectors, at SVD rtol
1e-12 or 1e-15. AAA trained on FEM values reaches the same floor. The floor is therefore not set by
basis truncation. The RB residual there is about 6e-10. The floor is consistent with the round-off
residual of the projected problem amplified by 1/σ_min(A), but **we have not isolated its cause
further**. Both floors are three orders of magnitude below the FEM discretisation error (1.4e-4).
They are irrelevant to the physics, but they bound what "agreement with the FEM" can mean.

## 6. Tests that pin this down (fast, in CI)

`tests/phase7/test_phase7.py`:

- definition and decay of the potential;
- second-order convergence to the ODE, with flux conservation;
- convergence to the exact S-matrix for rectangular barriers on the shared mesh, and refusal of
  off-mesh barrier edges;
- complex-E continuation equal to the real-axis solution on the axis, and the narrow pole at 0.95846;
- σ_min equal to a dense SVD, with the certified value below it.
