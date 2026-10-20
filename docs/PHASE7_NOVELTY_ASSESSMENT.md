# Phase 7 novelty assessment

**Bottom line.** Phase 7 makes **no discovery and introduces no new method**. Each method it uses
is published, and each qualitative finding is either known or expected from known theory. What
Phase 7 adds is coursework-level work: an independently validated benchmark on a smooth potential,
plus a leakage-free comparison of methods at equal query budgets on that benchmark. We found no
identical study, but our search could not read full texts, so this is not a claim of priority.

**How the literature was checked.** A search on 2026-10-20 used web search only. arXiv, APS, OSTI
and publisher full texts were not reachable from this environment. Titles, authors, venues and
abstracts were confirmed from indexed pages; the full texts of the closest works were **not
read**. Items marked [snippet] rest on search snippets only. The Phase 6 review
(`PHASE6_LITERATURE_REVIEW.md`) covers further nuclear-physics emulator work.

## 1. Prior work closest to each Phase 7 component

### Projection-based emulators for Schrödinger scattering (most important)

- **Maldonado, Drischler, Furnstahl, Mlinarić**, "Greedy emulators for nuclear two-body
  scattering", *Phys. Rev. C* 112, 024002 (2025), arXiv:2504.06092. Galerkin and LSPG reduced
  models with reduced-space residual estimators and greedy snapshot selection. Greedy needs far
  fewer snapshots than POD. **This is the direct precedent** for our RB-greedy-vs-POD results. Its
  parameters appear to be potential strengths [snippet: whether energy is also a parameter was not
  verified].
- **Giri, Kim, Drischler, Elster, Furnstahl**, "Active learning emulators for nuclear two-body
  scattering in momentum space", *Phys. Rev. C* 113, 044001 (2026), arXiv:2512.17842. Greedy
  (Petrov–)Galerkin emulators of the t-matrix with error estimates.
- **Furnstahl, Garcia, Millican, Zhang**, *Phys. Lett. B* 809, 135719 (2020).
- **Melendez, Drischler, Garcia, Furnstahl, Zhang**, *Phys. Lett. B* 821 (2021).
- **Drischler, Quinonez, Giuliani, Lovell, Nunes**, *Phys. Lett. B* 823, 136777 (2021). This one
  covers Kohn anomalies as a failure mode.
- **Odell et al.**, "ROSE", *Phys. Rev. C* 109, 044612 (2024).
- Review: **Duguet, Ekström, Furnstahl, König, Lee**, *Rev. Mod. Phys.* 96, 031002 (2024).
- Resonances with emulators:
  - **Yapa, Fossez, König**, *Phys. Rev. C* 107, 064316 (2023);
  - **X. Zhang**, arXiv:2411.06712.

### RB for wave problems with resonances

- **Hesthaven, Rozza, Stamm**, *Certified Reduced Basis Methods for Parametrized PDEs*, Springer
  (2016).
- **Chen, Hesthaven, Maday, Rodríguez**, *SIAM J. Sci. Comput.* 32, 970 (2010). Harmonic Maxwell.
- **Fares, Hesthaven, Maday, Stamm**, *J. Comput. Phys.* (2011). The EFIE with the wavenumber as
  parameter. The inf-sup constant can approach zero.

### Certified bounds near resonances

- **Sen, Veroy, Huynh, Deparis, Nguyen, Patera**, *J. Comput. Phys.* 217, 37 (2006). The Helmholtz
  bounds "become singular as we approach resonance".
- **Huynh, Rozza, Sen, Patera**, *C. R. Math.* 345, 473 (2007). The successive constraint method
  (SCM).
- **Huynh, Knezevic, Chen, Hesthaven, Patera**, *CMAME* 199, 1963 (2010). Natural-norm SCM.
- **Hess, Grundel, Benner**, *IEEE Trans. MTT* 63, 3549 (2015).
- **Sirković, Kressner**, *SIAM J. Matrix Anal. Appl.* 37, 695 (2016).
- The augmented-matrix formulation of singular values: **Golub, Kahan**, *J. SIAM Ser. B* 2, 205
  (1965).

### Rational approximation vs projection; poles from data

- **Nakatsukasa, Sète, Trefethen**, AAA, *SIAM J. Sci. Comput.* (2018).
- **Bonizzoni, Nobile, Perugia, Pradovera**, least-squares Padé of Helmholtz maps, *Adv. Comput.
  Math.* 46 (2020).
- **Pradovera**, minimal rational interpolation (MRI), *SIAM J. Numer. Anal.* (2020).
- **Pradovera**, greedy Loewner, *Adv. Comput. Math.* (2023).
- **Nobile, Pradovera**, *ESAIM: M2AN* 55, 1895 (2021). Rational frequency surrogates extended to
  more parameters by pole matching, with documented difficulties.
- **Bruno, Santana, Trefethen**, AAA for resonances, *SIAM J. Sci. Comput.* 48, A1260.
- **Bruschini**, "SPARSE", arXiv:2507.01002. AAA on real-energy T-matrix elements of radial
  Schrödinger equations to locate poles [snippet].
- Loewner and moment matching: **Mayo, Antoulas** (2007); **Schulze, Unger, Beattie, Gugercin**,
  *Linear Algebra Appl.* 537 (2018).

### Neural surrogates

- **Hesthaven, Ubbiali**, POD-NN, *J. Comput. Phys.* 363 (2018). This is our POD-regression
  baseline family.
- **Komp, Valleau**, *J. Phys. Chem. A* 124, 8607 (2020). A neural network for quantum rate
  constants through 1D single and double barriers.
- **Rahaman et al.**, spectral bias, ICML 2019.

### Basis transfer

- **Amsallem, Farhat**, *AIAA J.* 46, 1803 (2008). ROM adaptation.
- Enrichment by a few greedy steps on a new problem is common practice.

## 2. Classification of the Phase 7 findings

| Finding (see `PHASE7_RESULTS.md`) | Status | Why |
|---|---|---|
| FEM converges O(h²) to an independent ODE solution; resonances are mesh-, domain- and grid-stable poles | **Standard** (verification, not research) | textbook FEM theory |
| Galerkin RB reaches max abs Δt ≈ 1e-9 in 3D (E, V_L, V_R) with 48 snapshots, while POD regression, RBF and an MLP (up to 4096 points) stay at ≈ 1 and resolve no held-out resonance | **Known** | Maldonado et al. 2025; Giri et al. 2026; general RB literature. We confirm it on a different potential family |
| Along one energy line, RB reaches the floor with 16 queries, adaptive AAA with 24 and uniform AAA with about 48 (same order) | **Expected / known** | with exact transparent BCs the operator is polynomial in k, so Galerkin RB on a line is itself a rational function of k. Rational interpolation and moment matching are known to be closely related (MRI, Loewner). Not surprising |
| Direct interpolation of t generalises poorly across heights while RB does not | **Known in kind** | consistent with the pole-matching difficulties of Nobile & Pradovera 2021 and with MOR folklore |
| Basis size tracks the resonance count (≈ 1.5–3 per resonance at fixed widths) much more than resonance width (Γ_min × 160 smaller → ≤ 1 extra function at 1e-4) | **Expected; mild confirmation** | consistent with Phase 6 E4, and with n-width / meromorphic-structure arguments; we found no paper with this exact deconfounded sweep |
| The attainable RB accuracy in double precision scales as ≈ 5e-11 / Γ_min across 19 geometries | **Expected** | forward error ≈ eps × condition number, with σ_min ∝ Γ near a resonance. A useful quantitative illustration, not a new result |
| Certified bound res/σ_min holds on every point and is loose near resonances | **Known** | Sen et al. 2006; SCM literature. Our σ_min is computed exactly at full order, so the bound is an a-posteriori certificate, **not** an online-efficient certified RB bound |
| Power-iteration ‖A⁻¹‖ is only a lower estimate | **Standard** | elementary numerical linear algebra |
| A rectangular-barrier basis (48 snapshots, no Gaussian data) reaches the FEM discretisation error on Gaussians, at about twice the native basis size; 8 Gaussian snapshots take it to 1.5e-7 | **Not established as new; one test** | an empirical observation for one pair of families sharing one mesh |
| Greedy sampling is not uniformly better than a Sobol design in 3D (similar max error; better resonance recall at small n) | **Consistent with Phase 6** | Phase 6 found the same; no claim |
| Noise sensitivity of AAA on uniform grids (Froissart doublets, spurious poles); adaptive AAA is more noise-tolerant than RB with noisy snapshots | **Known in kind** | AAA literature, Bruno–Santana–Trefethen |
| Near-degenerate doublets resolved by RB and adaptive AAA | **Expected** | both represent poles; no claim |

## 3. What can be claimed (coursework level)

1. An **independently validated** smooth-potential benchmark for 1D resonant tunnelling. It has an
   ODE reference, a convergence study, and an artefact analysis of the resonances, with
   reproducible configurations and tests.
2. A **leakage-controlled comparison at equal query budgets** of projection RB (Galerkin, LSPG,
   greedy and a-priori designs), POD regression, RBF, an MLP, AAA (uniform and adaptive) and PCHIP
   on that benchmark. Methods are separated by the information they use. In-domain and
   out-of-domain errors and resonance-level metrics are reported separately.
3. A **deconfounded basis-size study**: count versus width versus energy window. Combined with the
   1/Γ round-off floor, it explains why some tolerances are unreachable in double precision.
4. Because the problem is 1D, the exact σ_min is affordable. That makes it possible to measure the
   **effectivity of a certified bound directly**, rather than through an SCM lower bound.

## 4. Claims that must not be made

- That RB or emulators for quantum scattering are new, or that greedy residual-driven emulators
  are new (Maldonado et al. 2025).
- That AAA pole extraction for scattering, or the AAA ≈ RB parity on an energy line, is new or
  unexplained.
- That the σ_min bound is an online-efficient certified RB method.
- That MLPs are inferior in general. We tested one standard architecture on small data.
- That RB transfers or extrapolates across potential families in general, on the basis of one
  rectangular → Gaussian test and two out-of-domain boxes.
- That RB is a speed-up over the FEM for few queries. Break-even needs thousands of queries once
  the greedy is counted (`results/phase7/cost/cost.md`).

## 5. Before any publication

Read in full: Maldonado et al. 2025, Giri et al. 2026, Bruschini (SPARSE), Pradovera 2020 (MRI),
Nobile & Pradovera 2021, and Sen et al. 2006. Check in particular whether energy is a parameter in
the nuclear greedy emulators. None of this is needed for the coursework.
