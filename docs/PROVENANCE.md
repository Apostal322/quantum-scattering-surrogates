# Provenance: published algorithms, our implementation choices, and new empirical findings

This file separates three things that are easy to blur in a report:

* **(P) Published.** Algorithms and facts taken from the literature or textbooks.
* **(I) Implementation.** Choices made in this project: parameters, adaptations, glue code. They are
  defensible, but they are not results.
* **(F) Findings.** New empirical observations made here. They are valid only for the systems tested
  and should not be read as general theorems.

## 1. Physics and numerics

| item | type | source / note |
|---|---|---|
| Stationary Schrödinger equation, amplitudes t and r, T + R = 1, rectangular-barrier formula | P | textbooks (Landau–Lifshitz QM; Griffiths) |
| Breit–Wigner form; resonances as poles of the S-matrix at E_r − iΓ/2 | P | scattering theory (Taylor 1972; Landau–Lifshitz) |
| Redheffer star product of scattering matrices | P | Redheffer (1962); its use for stable multilayer tunnelling, e.g. Ko & Inkson (1988) |
| Per-layer propagator written in entire functions of q², with sin(qd)/q = d·sinc | I | Our formulation; it avoids a branch choice and the turning-point degeneracy |
| Analytic continuation to complex E; Newton refinement of 1/t(E) for exact poles | I | Standard numerical technique, our implementation |
| DOP853 ODE cross-check, integrated right to left | P (method) / I (use) | Hairer–Nørsett–Wanner; `scipy.integrate.solve_ivp` |
| Staircase approximation of a smooth potential: O(h²) convergence measured (1.0e-4 → 4.0e-7 for 500 → 8000 layers) | I + F | Our measurement (Experiment 5 setup) |

## 2. Surrogate and sampling methods

| item | type | source / note |
|---|---|---|
| Linear, PCHIP, cubic spline | P | Fritsch & Carlson (1980); de Boor (1978); SciPy |
| Gaussian-process regression, marginal-likelihood kernel selection | P | Rasmussen & Williams (2006); scikit-learn |
| GP length-scale lower bound of half the mean sample spacing | I | Our fix for a degenerate likelihood optimum we observed (F) |
| AAA rational approximation | P | Nakatsukasa, Sète & Trefethen (2018); `scipy.interpolate.AAA` |
| Approximating t as a function of **k** rather than E, and approximating the **complex amplitude** rather than T | I | Motivated by the textbook analytic structure (t meromorphic in k). The choice is ours. |
| Discarding Froissart doublets (negligible residue, or pole–zero pairs) | P (concept) / I (thresholds) | Spurious poles are discussed in the AAA paper; our thresholds are 1e-10·max\|t\| and 1e-3·\|Im pole\| |
| Greedy sampling at argmin \|barycentric denominator\| with a look-ahead error check | P (idea) / I (adaptation) | The greedy MRI loop (gMRI) in the code of Pradovera, Hiptmair & Perugia. Their MRI weights are not determined for scalar outputs, so we use AAA least-squares weights. Our parameters: tolerance 1e-8, three consecutive passes, then bisection of the largest gap. |
| Calibrated error estimate driving a greedy choice | P (principle) | Greedy emulators of Giri et al. (2026). In Phases 1–5 we use only the principle, because the scalar-output setting provides no solution vectors; Phase 6 implements the residual/Galerkin variant with FEM wave functions. |
| Segment-length adaptive sampling √(ΔÊ² + Δy²) | P | Default loss of 1D adaptive-sampling libraries (`adaptive`, Nijholt et al.) |
| Segment sampling on the complex amplitude, or on log T | I | Our variants |
| MLP, Adam, cosine schedule, low-data protocol (interleaved validation, early stopping with refit) | P (components) / I (protocol) | Kingma & Ba; PyTorch. The protocol and its ablation are ours. |
| Fourier features, with σ selected on validation data | P (method) / I (selection) | Tancik et al. (2020) |
| Deep-ensemble disagreement as an acquisition function | P (method) / I (use) | Lakshminarayanan et al. (2017) |
| POD + MLP of reduced coefficients; compression error reported separately from coefficient error | P (method) / I (setup) | Hesthaven & Ubbiali (2018); the error separation follows the spirit of Dölz & Henríquez |
| Dissipative pretraining with an absorbing well −iW (W = 0.05), half the budget lossy and half lossless, same geometries | P (idea) / I (transfer to 1D) | Nam, Park & Jang (2026). They broaden electromagnetic resonances with material loss; the transfer to 1D and the value of W are ours. |
| Hybrid surrogate: learn the smooth single-barrier S-matrix, compose exactly with the star product | I | Our structure-exploiting model. It requires knowing the block decomposition. |
| **Phase 6:** P1 finite elements with exact transparent (Robin) scattering boundary conditions, affine in (E, k, V_j) | P (method) / I (implementation) | Textbook FEM; validated here against the S-matrix solver (O(h²), same global phase) |
| **Phase 6:** POD/SVD basis, Galerkin and LSPG projection, offline/online splitting, residual norm, ‖e‖ ≤ ‖A⁻¹‖‖res‖, weak greedy | P | Hesthaven, Rozza & Stamm (2016); Sen et al. (2006); scattering emulators of Drischler et al. (2023), Gnech et al. (arXiv:2511.01844), Catacora-Rios et al. (PRC 113, 044623) |
| **Phase 6:** residual evaluated through a thin QR of the residual space (instead of Gram matrices) | P (standard numerical linear algebra) / I (use) | Our fix after the Gram form showed a ~1e-7 round-off floor that stalled the greedy |
| **Phase 6:** POD coefficient regression and direct t interpolation on the same FOM solves, as comparators | I (protocol) | Equal FOM-solve budgets; the "best design" for these baselines is chosen on test data (optimistic for them) |

## 3. New empirical findings (F), with their scope

| finding | evidence | scope / caveat |
|---|---|---|
| Local methods on uniform grids miss resonances narrower than the grid spacing | Exp. 1–2 | Expected behaviour; quantified here |
| AAA on t(k) recovers every resonance from 15–30 uniform evaluations | Exp. 3 (38 systems), Exp. 5 (unseen families) | Piecewise-constant and smooth finite-range 1D potentials; exact or low-noise data |
| The amplitude is a better AAA target than T (about half as many points) | Exp. 1, 3 | Consistent with the pole count of \|t\|²; observed, not proven in general |
| Greedy look-ahead self-certification gave no false certification | Exp. 3 (38/38), Exp. 5 | Empirical; a theoretical guarantee would need error bounds we do not have |
| Detectability limit: Γ/h ≳ C·max(data error, background-approximation error) | Exp. 3b map | Heuristic; C is not universal |
| Segment sampling on t beats segment sampling on T for local interpolants | Exp. 2–3 | Convention-dependent; disappears for phase reference X = 50 |
| Stationary GPs degrade on multi-scale adaptive designs; MLPs miss narrow peaks even with data (spectral bias) | Exp. 2 | Small models, our protocols |
| Hybrid S-matrix composition is the most sample-efficient parametric surrogate | Exp. 4 | Requires a known decomposition; 1D |
| AAA result holds on 27 unseen systems (asymmetric, doublets, smooth potential, noise); limits: under-resolution with many poles, instability only in under-resolved fits, premature self-certification at noise 1e-4 | Exp. 5 | Same scope as above |
| Pole width vs FWHM: agree to 8e-5 for isolated resonances, deviation ~ (Γ/Λ)², fails for overlapping doublets | Exp. 5, §10.9 | Standard Breit–Wigner/Fano physics; quantified here |
| Solver-call reduction becomes wall-clock speed-up only with an expensive solver (670–700× with the ODE solver, none with the fast one) | `results/cost/` (two runs) | Single machine, timing spread ~30–50 % |
| **Phase 6:** projection RB reaches the FOM round-off floor with 10 FOM solves in E (AAA: 20 with its own greedy) and 24 in (E, V_L, V_R), where RBF and POD regression fail up to 64 | `results/phase6/SUMMARY.md` | One 1D double-barrier family, affine parameters only; RB is intrusive |
| **Phase 6:** residual ranks wave-function errors (Spearman 0.84–0.99) but the bound is loose (median 440–1100×) | `results/phase6/rb_3d/` | Euclidean nodal norms; ‖A⁻¹‖ by power iteration (a lower estimate) |
| **Phase 6:** basis size nearly independent of resonance width, growing with the number of resonances | `results/phase6/nwidth/` | 9 configurations; number of resonances and well length are confounded; hypothesis, not theorem |
| **Phase 6:** RB online cost vs FOM and exact solver | `results/phase6/cost/cost.md` | RB beats the FEM FOM per query; it does not beat the vectorised exact solver; break-even counts in the file |

**Established methods, not ours:** AAA and rational approximation in general, splines, PCHIP, Gaussian processes,
MLPs, Fourier features, POD, the Redheffer star product. The project contributes a systematic comparison, the
choice of a physically motivated representation (t as a function of k), measured limits, and cost accounting.

## 4. Reference verification log

The network policy of the execution environment blocked arxiv.org, doi.org, publisher sites, Crossref, OpenAlex
and Semantic Scholar (all requests timed out). Only github.com was reachable. Verification was therefore limited
to what the authors state in their own repositories:

| reference | what was verified | how |
|---|---|---|
| Pradovera, Hiptmair, Perugia. "Surrogate modeling of resonant behavior in scattering problems through adaptive rational approximation and sketching", J. Sci. Comput. (2025) | title, authors, year, DOI 10.1007/s10915-025-03020-x, preprint arXiv:2503.10194 | README of github.com/pradovera/hybridScatteringAmplification |
| Giri, Kim, Drischler, Elster, Furnstahl. "Active learning emulators for nuclear two-body scattering in momentum space", Phys. Rev. C 113, 044001 (2026) | full citation, DOI 10.1103/s6my-pqs9, arXiv:2512.17842 | BibTeX in the README of github.com/buqeye/ms_greedy_emulator |
| Nam, Park, Jang. "Data-efficient electromagnetic surrogate solver through dissipative relaxation transfer learning", Adv. Opt. Mater. (2026), DOI 10.1002/adom.71366 | title and arXiv:2601.18235 only | The repository README says "currently under review in Advanced Optical Materials". **The DOI and the publication status could not be verified.** |
| Dölz, Henríquez. "Fully discrete analysis of the Galerkin POD neural network approximation with application to 3D acoustic wave scattering", arXiv:2502.01859 | nothing | No reachable source. **Unverified**; cited as supplied. |
| Textbook and classic references (AAA 2018, Fritsch–Carlson 1980, Tancik 2020, …) | not checked online | Cited from established bibliographic knowledge; check against the originals before submission |

**The full texts of the four recent papers were not read** in this environment. The methods were implemented from
the authors' public code and abstracts (summarised in the session log), and the reports do not attribute to these
papers any result they were not checked to contain.
