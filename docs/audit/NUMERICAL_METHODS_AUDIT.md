# Numerical methods audit (audit, 2026-10-23)

**Verdict.** The solvers, the reduced-basis projection and the pole finders are correct as
implemented. Two code paths could silently return unconverged results; both are fixed and covered
by tests:

- `solve_ode_smooth` ignored integrator failure (P2-ODE);
- the surrogate pole search returned a non-root (P2-POLE).

The main correction is about meaning, not computation. The residual/σ_min "certified" bound is
mathematically valid, but it is not a computer-verified certificate, and evaluating it costs
thousands of FEM solves (P1-CERT).

## 1. Scattering-matrix solver (`physics/smatrix.py`)

| Item | Finding |
|---|---|
| Layer propagator | [[cos qd, sin(qd)/q], [−q sin qd, cos qd]] written with `d·sinc(qd/π)`: entire in q², so it needs no branch and is regular at E = V |
| Layer S-matrix | derived from matching (1 + r, ik(1 − r)) → M → (t, ikt); symmetric single layer, so the left and right amplitudes coincide (correct for one homogeneous layer) |
| Composition order | Redheffer star product with A on the left: t = t_B (1 − r'_A r_B)⁻¹ t_A, r = r_A + t'_A r_B (1 − r'_A r_B)⁻¹ t_A. Agrees with an independent global matching system to 5e-15 (C1) |
| Evanescent layers | only bounded quantities are combined across layers; a guard refuses κd > 600 (`test_rejects_invalid_input`). κa = 50 checked against the matching system: 4e-16 relative |
| Vectorised vs scalar | identical to the last bit on 2001 energies |
| Complex E | `transmission_complex_energy`, principal √E; matches the matching system at complex E to ≤ 4e-12 relative near poles |

## 2. ODE reference (`physics/ode_solver.py`)

- It integrates from the transmitted wave e^{ikx} on the right towards the left. That direction
  follows the growing solution inside barriers, which is the stable choice for this shooting
  problem. It then decomposes the solution at x_L into e^{±ikx}.
- DOP853 with rtol 1e-11 to 1e-12. Tolerance sensitivity was measured in Phase 7 A1: changing
  rtol from 1e-12 to 1e-11 changes t by ≤ 6e-11, and the flux defect is ≤ 5e-13.
- **Defect fixed (P2-ODE).** `solve_ode_smooth`, the Phase 7 reference, did not check
  `sol.success` and accepted E ≤ 0. A failed integration would have returned the last state as if
  it were a solution. It now raises. This was found by code inspection, and no archived result was
  affected: every Phase 7 ODE call in the validation was self-consistent, so it converged.

## 3. Finite-element models (`phase6/fom.py`, `phase7/fom.py`)

- **Weak form.** ∫ψ'φ' + ∫Vψφ − E∫ψφ − ik[ψφ]_{boundary} = −2ik φ(x_L) (incident wave normalised
  at x_L). P1 elements are used on a uniform mesh, and the matrices are assembled element by
  element.
- **Potential matrices.** 4-point Gauss–Legendre quadrature per element. It is exact for indicator
  potentials whose edges are mesh nodes; this is enforced, and off-node edges raise. For Gaussians
  the quadrature error is O(h⁸).
- **Convergence.** O(h²) against the ODE (ratio 3.93–4.01) and against the exact and matching
  solvers for rectangles.
  - The dominant error is numerical **dispersion**: with V = 0, t − 1 = k³h²L/24 and |t| = 1
    exactly (C3, new test).
  - Domain truncation is negligible: ≤ 3e-13 at rel_cut = 1e-14.
- **Linear solves.** Tridiagonal complex LU (`solve_banded`). The condition number near the
  narrowest resonance is ≈ 4e7, and two backward-stable solvers then agree only to 6e-9. That is
  the FEM's own round-off floor.
- **Affine decomposition.** A(μ) = A₀ − E·M − k·(iB) + Σ V_j M_j and b = −2ik e₀ (`test_affine_terms_reassemble_the_operator`).

## 4. Reduced-basis projection (`phase6/rb.py`, used in Phases 6–7)

| Item | Finding |
|---|---|
| Snapshot orientation | snapshots are columns (N × n); POD/SVD via `np.linalg.svd(full_matrices=False)`, left singular vectors kept above rtol·s₁ (rtol = 1e-12) |
| Inner product | Hermitian: A_r = Vᴴ A V, f_r = Vᴴ f. With a complete orthonormal basis, Galerkin and LSPG reproduce the FOM to 1e-10, also at complex E (new test) |
| LSPG | min ‖b − A V c‖ solved by batched QR of the thin-QR residual coordinates (no normal equations) |
| Residual | computed in a thin-QR basis of span[f, A_q V]; agrees with the directly assembled full-order residual to round-off (|Δ| ≤ 1e-4 relative + 10·eps·‖A‖‖ψ̂‖; new test). Near the floor (‖res‖ ~ 1e-12) the two differ by up to 8 %, as expected from round-off |
| Observable extraction | t = (Vc)(x_R)·e^{−ikL}, r = ((Vc)(x_L) − 1)·e^{2ikx_L}, the same function as the FOM |
| Reduced conditioning | cond(A_r) ≤ 5e6 on the test scans; complex64 online solves lose 1–2 digits (Phase 7 F) |
| Distinct methods | Galerkin, LSPG, POD regression (coefficients interpolated, **no operator**; new test asserting it never touches A) and direct t interpolation are separate code paths |

## 5. The residual / σ_min error bound

**Mathematics.** Let ψ̂ = Vc and r = b − Aψ̂. Then e = ψ − ψ̂ = A⁻¹r. For the Euclidean norm on
nodal vectors this gives ‖e‖₂ ≤ ‖A⁻¹‖₂‖r‖₂ = ‖r‖₂/σ_min(A).

- Dimensions: A is N × N complex and invertible for real E > 0 (unitary scattering) and away from
  poles.
- The sign of r does not matter for the norm.
- Observable: |t̂ − t| = |e_N| ≤ ‖e‖_∞ ≤ ‖e‖₂ on the real axis, where |e^{−ikL}| = 1.
- The bound concerns the **discrete** solution only. It says nothing about the FEM's own error
  against the continuum, nor directly about errors in E_r or Γ.

**How σ_min is computed.** `AffineFOM.sigma_min` takes the smallest non-negative eigenvalue of the
Hermitian augmented matrix [[0, A], [Aᴴ, 0]]. Interleaved, it is banded with half-bandwidth 3, and
LAPACK banded bisection is used. In the brief's classification this is **an exact singular value up
to numerical precision**: not a Ritz value, and not an iterative estimate.

- Against dense SVD it is accurate to ≤ 1.4e-8 relative, including at a pole with σ = 3.5e-6.
- `certified=True` subtracts 2m·eps·‖A‖₁. That is a heuristic choice of LAPACK's unspecified
  backward-error constant p(m), not a proof.

**Status.** The bound is exact in exact arithmetic and evaluated in floating point. Neither the
eigenvalue (heuristic margin) nor the residual (no rounding analysis) is verified with interval
arithmetic. Empirically it held at every one of 326 points × 4 bases in Phase 7 E, and down to
the round-off floor with bases of 24–60 functions (errors down to 3e-11), with a margin of at least
22× (`test_residual_sigma_min_bound_holds_down_to_the_round_off_floor`). The documents now call it
an "a-posteriori bound evaluated in floating point". The data keys keep the name `certified` for
compatibility.

**Cost.** One evaluation takes ≈ 2.2 s at N = 4331 on the audit machine, i.e. ≈ 3,800 FEM solves
(`results/audit_2026/timing.json`). It is a validation instrument. An online-efficient certified RB
bound would need an offline lower bound of the stability constant (e.g. SCM), which was not built.

**Phase 6 power iteration.** `phase6/baselines.inverse_norm` runs 30 power iterations on
(AᴴA)⁻¹. That is an uncertified iterative estimate that approaches ‖A⁻¹‖ from below: the ratio to
the exact value is 0.966–0.999998, and a new test asserts it never exceeds 1/σ_min. The Phase 6
"bound" therefore was not a bound; it held empirically. `REPORT_RU.md` §10.11 called it "строгая"
(rigorous); that word is removed and an erratum added.

## 6. Pole finding

| Finder | Method | Acceptance | Checked |
|---|---|---|---|
| `physics/poles.refine_pole` (Phases 1–5) | Newton on the exact 1/t(E), central-difference derivative, from AAA candidates | step < 1e-14·\|z\| | Stage B poles reproduced (CI); dedup 1e-9 |
| `phase7/fom.find_pole` | secant on the FEM 1/t, keeps the best iterate | Im z < 0, Re z > 0 and \|1/t(z)\| < 1e-5·\|1/t(Re z)\| | 6 poles agree with an independent staircase S-matrix to ≤ 3e-6 Γ (C6) |
| `phase7/resonances.find_poles` | windowed AAA (window 1.5, 1500 points) for candidates | poles with Im k < 0 and \|residue\| > 1e-12; duplicates within 1e-6 removed | pole counts stable over 6 meshes and domain change (Phase 7 A) |
| `phase7/slice1d.secant_pole` (surrogates) | secant on a surrogate's 1/t | **was:** none; **now:** same test as `find_pole` | new test |

Risks covered: wrong branch (principal √E is correct for Re E > 0), spurious AAA poles (only
candidates; every pole is verified on the exact or FEM amplitude), duplicates (deduplication),
false convergence (acceptance test). Not covered: poles with Re E ≤ 0, and very broad poles far
from the axis. No reported result involves them.

## 7. Resonance metrics

- **Peak/FWHM** (`physics/resonances.find_resonances`, `phase6/experiments.half_max_width`):
  - peaks are refined by bounded maximisation, and the half-maximum crossings by Brent;
  - the search never leaves the scan (regression test for the earlier E ≤ 0 bug, plus a new
    near-threshold test);
  - the width is undefined (`None`/NaN) when the half maximum is not reached;
  - overlapping peaks merge into one, with an FWHM that is neither pole width (new test).
- **"Resolved"** (Phases 5–7): max |T̂ − T| ≤ 0.1 T_peak over ±2Γ. A single good point near the peak
  does not count (new test).
- **Experiment 4** used a weaker rule, "T̂ > 0.5 at one point within ±Γ", which overstated recovery
  (P1-EXP4-METRIC; re-scored).
- **Spurious peaks:** prominence > 0.05 and farther than 3Γ from every pole. Noise below the
  prominence threshold is ignored (new test).
- **Units:** width errors are reported as relative errors (FWHM_rel), and peak shifts in units of Γ.
