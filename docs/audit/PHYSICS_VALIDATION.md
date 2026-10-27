# Physics validation (audit, 2026-10-23)

**Verdict.** No physics error was found in any solver. Equations, boundary conditions, phase
conventions, branch choices and resonance definitions are consistent across the S-matrix, ODE,
Phase 6 FEM and Phase 7 FEM solvers. Every solver was checked against at least one reference
that shares none of its algorithm:

- a new global matching solver;
- the analytic Born limit;
- exact metamorphic relations;
- an independent pole computation.

One property of the Phase 7 FEM was previously undocumented, though it is not an error: its error
is dominated by numerical dispersion, which grows like k³h²L.

Evidence: `results/audit_2026/physics_checks.{json,md}` (`scripts/audit/independent_physics_checks.py`)
and `tests/audit/test_audit_regressions.py`.

## 1. Model and conventions

- Equation: −ψ'' + V(x)ψ = Eψ (ħ²/2m = 1), k = √E, V = 0 outside a finite interval. The
  equation is the same in `smatrix.py`, `ode_solver.py`, `phase6/fom.py` and `phase7/fom.py`.
  For the FEM this was checked through its weak form: stiffness − E·mass + V-mass − ik·(boundary
  mass).
- Global amplitudes: ψ = e^{ikx} + r e^{−ikx} left of the potential and ψ = t e^{ikx} right of it.
  All solvers return amplitudes in this one coordinate:
  - the S-matrix solver converts from layer-local references (`t_loc e^{−ik(x_R−x_L)}`, `r_loc e^{2ikx_L}`);
  - the FEM imposes the incident wave e^{ik(x − x_L)} and converts in the same way;
  - the ODE solver decomposes the solution at x_L into e^{±ikx}.
  Agreement of r between solvers to 1e-15 (S-matrix vs matching), and of the FEM r phase with the
  analytic Born formula, confirms this.
- Transparent boundary conditions: ψ' = ikψ at x_R and ψ' = −ikψ + 2ik e^{ikx_L} at x_L, exact
  for V = 0 outside. The domain of the smooth Gaussian is truncated where V/V_max = 1e-14. The ODE
  shows the truncation changes t by ≤ 3e-13.
- Branches: the layer formulas use only even functions of q = √(E − V), so the S-matrix solver
  needs no branch choice and is regular at E = V. The asymptotic k = √E uses the principal branch.
  It is continuous across the positive real E axis, so E = E_r − iε reaches the second sheet
  (Im k < 0), where resonance poles lie. The validated continuation domain is Re E > 0 near the
  real axis. Poles with Re E < 0 or very large |Im E| are outside it, and no reported pole is there.

## 2. Checks

| ID | Check | Solvers | Reference (independent of) | Result |
|---|---|---|---|---|
| C1 | random multilayers (40 systems, 1–8 layers, V ∈ [−5, 30]), opaque barrier κa = 50, complex E | Redheffer S-matrix | global matching system, dense LU (`physics/matching.py`) | max \|Δt\|, \|Δr\| = 4.8e-15; opaque rel. 4e-16; complex E rel. ≤ 4e-12 |
| C2 | rectangular barriers on the shared FEM mesh | Phase 7 FEM | matching system | O(h²): error ratio 4.0 per halving |
| C3 | weak coupling V → λV, λ = 1e-5, asymmetric Gaussians | ODE, staircase S-matrix, FEM | analytic first-Born (t − 1)/λ, r/λ | ODE and staircase 1e-5 relative (= second-order Born term); FEM 9.3e-5 → 2.5e-5 for h = 0.005 → 0.0025, after subtracting its own V = 0 solution |
| C4 | translation by a = 0.37, mirror x → −x | ODE, FEM | exact relations t' = t, r' = r e^{2ika}; t and \|r\| mirror-invariant | ≤ 2e-13 (ODE), 4e-12 (FEM mirror) |
| C5 | Richardson FEM vs ODE, two height pairs | FEM | ODE (no mesh) | 1.6e-8 and 3.9e-9, against FEM errors of 3.5e-5 and 1.1e-5 at h = 0.00125 |
| C6 | 6 resonance poles (Γ from 1.3e-4 to 0.22) | FEM (Richardson in h) | staircase S-matrix continued to complex E (Richardson in layers) | ΔE_r ≤ 3.1e-6 Γ, ΔΓ/Γ ≤ 1.9e-8; \|1/t\| at the pole 4e-14–7e-11 of its real-axis value |
| C7 | flux \|t\|² + \|r\|² = 1, three height pairs, E ∈ [0.05, 15]; absorbing V_L = 10 − 0.5i | FEM, staircase | — | defect ≤ 1.4e-12 for real V; deficit ≥ 0.011 with absorption |
| C8 | threshold E = 1e-2 … 1e-8 | FEM, S-matrix | t ∝ k (no zero-energy resonance) | \|t\|/k constant to 2e-4 |
| — | textbook single barrier, T + R = 1, ODE vs S-matrix, Stage B poles vs FWHM | S-matrix | `tests/test_physics.py`, `reproduce_key_results.py` | 17/17 headline numbers reproduced (clean clone, CI) |

**Tolerances.** Unitarity tolerances are set by round-off, which grows like N·eps: ≤ 1e-11 for
N ≈ 17 000 FEM nodes. Born-limit tolerances are set by the second-order Born term, of relative
size λ·V/E ≈ 1e-5. FEM tolerances are set by its measured O(h²) error.

## 3. Resonances, poles and widths

- Definitions are consistent. A resonance is a pole E_p = E_r − iΓ_p/2 of t on the second sheet.
  The peak of T and its FWHM are measured separately. Both are reported, and the report explains
  why they differ: FWHM/Γ_p − 1 = O((Γ/Λ)²) for isolated resonances; Fano asymmetry; a single
  merged peak for overlapping doublets (`REPORT_RU.md` §3.4, §10.9).
- Pole finders:
  - **Phases 1–5:** `physics/poles.py` takes AAA candidates and refines them by Newton on the exact
    continued 1/t.
  - **Phase 7:** `phase7/fom.find_pole` uses secant iteration on the FEM 1/t. It is accepted only if
    |1/t(z)| < 1e-5·|1/t(Re z)| and Im z < 0. Candidates come from windowed AAA; duplicates are
    removed within 1e-6 relative.
  - **Surrogates:** `slice1d.secant_pole` returned its best iterate even when no root was reached.
    It now applies the same acceptance test (P2-POLE). The archived RB pole errors (≤ 2e-8 Γ,
    starting at half the true imaginary part) were genuine roots.
- Independence: the C6 comparison uses a staircase S-matrix and Newton refinement. It shares no
  rational approximation, discretisation or root finder with the FEM poles. AAA only supplies
  initial guesses in either method.
- Not every transmission maximum is treated as a pole. Resonance metrics start from FEM poles, and
  a resonance is scored only if the FEM peak of T within ±3Γ of the pole reaches 0.05 (`phase6/experiments.scan_metrics`).
  Asymmetric configurations with broad poles and low peaks are kept in the pole list but not
  scored as peaks.

## 4. Error relative to the FEM vs error relative to the continuum

The Phase 6 and Phase 7 surrogate errors are measured against the FEM at a fixed mesh. The FEM's
own error against the continuum problem is 4e-5 to 2e-4 in t at h = 0.0025:

- with V = 0 it equals the P1 dispersion error k³h²L/24 to four digits;
- t has modulus exactly 1, so the error is a phase error;
- it is largest at high E and for long domains.

So a surrogate error of 1e-9 "relative to the FEM" means agreement with a discrete model whose
physical accuracy is 1e-4. All Phase 7 documents now state this. Every surrogate also reaches a
round-off floor of 3–8e-8 near narrow resonances, and the floor scales as ≈ 5e-11/Γ_min (Phase 7 D).
That limit comes from conditioning, not from the reference solver.

## 5. Remaining physics limitations (not errors)

- One spatial dimension, one channel, finite-range potentials. Phase 6/7 "three-parameter"
  problems vary three parameters (E, V_L, V_R); they are not three-dimensional scattering.
- The Gaussian potential is solved truncated at relative height 1e-14. The truncation error is
  measured, not removed.
- Poles are validated for Re E > 0 near the real axis only.
