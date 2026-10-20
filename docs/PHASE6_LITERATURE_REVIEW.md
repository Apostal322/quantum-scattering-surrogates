# Phase 6 literature review (verified 2026-10-20)

**Scope and honesty statement.** The environment used for Phase 6 could not reach arxiv.org,
export.arxiv.org, zenodo.org or journal sites: DNS fails for all of them, and the fetch tool is
refused. GitHub is reachable only through an anonymous git proxy, used for one repository below.
Every statement here therefore rests on **one of three verification levels**, shown for each item:

| Level | Meaning |
|---|---|
| **V-code** | Repository cloned and inspected by us (files, commit, licence). |
| **V-index** | Title, authors, date and abstract-level content confirmed from search-engine indexes of arXiv/APS/SciPost pages (several independent result pages). **The full text was not read.** |
| **NOT FOUND** | The identifier could not be matched to any indexed document. We do not describe its content. |

Numerical results quoted below are **the authors' claims as stated in abstracts or index
snippets**. We did not reproduce them, except where Section 3 says so.

## 1. Primary sources

### [1] arXiv:2610.22129: "Bošković et al. (2026), CO2–He rovibrational cross sections". **NOT FOUND**

- The identifier did not resolve: arXiv is unreachable from this environment, and two searches
  returned no indexed document with this number. Title, authors, status and the stated
  supplementary data **could not be confirmed**.
- Closest **verified** prior work on the same system (V-index):
  - M. Selim, A. van der Avoird, G. C. Groenenboom, *State-to-state rovibrational transition rates for
    CO2 in the bend mode in collisions with He atoms*, arXiv:2309.03781 (2023). It reports rate
    coefficients to 500 K from coupled-channel (CC), coupled-states approximation (CSA) and VCC-IOS
    calculations. According to the indexed text, its supplementary material holds Figs. S1–S8 and
    a Fortran routine for the CO2–He potential. One index snippet also mentions partial-wave cross
    sections. Whether **machine-readable CC/CSA cross-section tables** exist could not be checked.
  - The same authors, *Efficient computational methods for rovibrational transition rates in molecular
    collisions*, arXiv:2206.04470 (2022): CSA variants and a nearest-neighbour Coriolis coupling
    (NNCC) method against full CC for CO2–He.
  - An APS March Meeting 2024 abstract (same group) states that CSA and VCC-IOS agree with CC "mostly
    to within 50 %" state to state and "within 20 %" for overall vibrational quenching rates.
- **Relevance:** this is exactly the CC-vs-CSA multi-fidelity setting of the Phase 6 brief. The data
  gate is documented in `PHASE6_DATA_AUDIT.md`.

### [2] arXiv:2511.01844: Gnech et al. (2025). **V-index**

- **Citation:** A. Gnech, X. Zhang, C. Drischler, R. J. Furnstahl, A. Grassi, A. Kievsky,
  L. E. Marcucci, M. Viviani, *Accurate and Efficient Emulation of Proton-Deuteron Scattering via the
  Reduced Basis Method and Active Learning*, submitted 3 Nov 2025 (nucl-th). Companion paper:
  arXiv:2511.10420, *... Detailed Description* (13 Nov 2025). Journal status not confirmed.
- **Physics:** proton–deuteron scattering below the deuteron breakup threshold, with high-fidelity
  hyperspherical-harmonic (HH) solutions. The varied parameters are three-nucleon-force low-energy
  constants of chiral EFT. The observables are phase shifts / S-matrix elements and cross sections.
- **Method:** reduced-basis emulators from (i) the Kohn variational principle and (ii) Galerkin
  projection of the HH linear system. Training points are chosen by greedy (active-learning)
  algorithms.
- **Adaptable here:** yes. Galerkin projection of an affinely parametrised linear scattering system,
  plus greedy selection, is exactly what Phase 6 implements for a 1D problem (Section 3).
- **Reproducibility limits:** no code or data repository was found in the indexed text, and the HH
  solver is not public to our knowledge.

### [3] arXiv:2510.13989: Jin et al. (2025), *Surrogate Models for Linear Response*. **V-index, V-code**

- **Citation:** L. Jin, A. Ravlić, P. Giuliani, K. Godbey, W. Nazarewicz (FRIB/MSU), arXiv dated
  15 Oct 2025. A FRIB news item states that it was published in Physical Review Research; the DOI was
  not confirmed.
- **Physics:** nuclear linear response in QRPA: E1 strength and the electric dipole polarisability
  of 180Yb, Gamow–Teller strength and the β-decay half-life of 80Ni. **This is not a scattering
  problem.**
- **Method:** a QRPA-structured reduced-order model and a parametric matrix model. The authors claim
  0.1–1 % accuracy and a 10^6–10^7 speed-up over QRPA solvers.
- **Code (V-code):** `github.com/ascsn/SMLR`, cloned at commit `6c1c259` (16 Apr 2026): 8447 files,
  including the high-fidelity QRPA strength files (`dipoles_data_all/`, `beta_decay_data_Ni_80/`) and
  notebooks. **There is no LICENSE file**, so redistribution and reuse rights are not granted; we copy
  nothing from it.
- **Adaptable here:** only conceptually (reduced-order structure + parametric matrices). The data are
  nuclear response functions, not scattering observables.

### [4] arXiv:2512.08097: Catacora-Rios et al. **V-index**

- **Citation:** M. Catacora-Rios, K. Beyer, P. Giuliani, K. Godbey, R. J. Furnstahl, F. M. Nunes,
  *Wavefunction-Based Emulation of Coupled-Channels Scattering with Non-Affinely Parametrized
  Interactions*, submitted 8 Dec 2025. The APS listing shows Phys. Rev. C 113, 044623 (27 Apr 2026),
  DOI 10.1103/tgf9-f2st.
- **Physics:** neutron elastic and inelastic scattering, n+48Ca (2+) and n+208Pb (3−), with
  rotational-model couplings.
- **Method:** an SVD basis of coupled-channel wave functions, (Petrov-)Galerkin projection, and
  empirical interpolation (EIM) for non-affine potentials.
- **Adaptable here:** the SVD-basis + Galerkin structure, yes. EIM is unnecessary in our problem
  because barrier heights enter affinely; it would be needed for moving edges (widths, positions).

### [5] arXiv:2610.03910: "Catacora-Rios et al. (2026), strongly deformed nuclei". **NOT FOUND**

Not indexed. The only related verified item is the APS DNP 2024 abstract *Coupled Channels Reduced
Order Scattering Emulators* (presenter M. Catacora-Rios; co-authors F. M. Nunes, P. Giuliani,
R. J. Furnstahl, K. Godbey), which may or may not correspond to it.

### [6] arXiv:2512.17687: Jin Lei, *Reduced basis emulator for elastic scattering in continuum-discretized coupled-channel calculations*. **V-index**

- **Details:** Tongji University; submitted 19 Dec 2025 (nucl-th). Journal status not confirmed.
- **Method:** POD of full CDCC solutions at sampled optical-potential parameters, then Galerkin
  projection.
- **Authors' claims:** d+58Ni at 21.6 MeV with 18 parameters emulated at once, elastic cross-section
  errors below 0.1 %, speed-up of about 10^2.
- **Adaptable here:** the POD + Galerkin pipeline, yes. The claimed speed-up is for CDCC solvers and
  says nothing about ours.

## 2. Secondary methodological references

| # | Identifier | Status | What is confirmed | Use in Phase 6 |
|---|---|---|---|---|
| 7 | arXiv:2610.11500 ("resonance-aware learning") | **NOT FOUND** | – | none |
| 8 | arXiv:2608.16152 | **V-index (excerpt only)** | Title: *Asymptotics-guided learning and symbolic regression for dispersive resonances*; authors not confirmed. Prediction = asymptotic formula + learned residual (ridge regression; symbolic regression), single resonator and dimer. | The "physics model + learned correction" idea underlies the additive-correction representation of the (blocked) multi-fidelity plan. Not implemented |
| 9 | arXiv:2412.12069 | **V-index** | H. Bahl, N. Elmer, L. Favaro, M. Haußmann, T. Plehn, R. Winterhalder, *Accurate Surrogate Amplitudes with Calibrated Uncertainties*, SciPost Phys. Core 8, 073 (2025). Heteroscedastic loss for systematic uncertainty, Bayesian NNs or repulsive ensembles for statistical uncertainty, coverage checked with pull distributions; gg→γγg loop amplitudes. | Coverage/pull methodology. Phase 6 uses the residual instead of a learned uncertainty, and checks it against true errors on held-out data |
| 10 | arXiv:2610.20978 (PCA–GPR) | **NOT FOUND** | – | none (PCA–GPR is generic; our POD-regression baseline plays that role) |
| 11 | arXiv:2609.03004 (data efficiency) | **NOT FOUND** | – | none |
| 12 | arXiv:2610.12632 (derivative-aware spectral learning) | **NOT FOUND** | – | none |

## 3. Background used for the novelty assessment

Status: the four items cited in the coursework report ([27], [31]–[33] there) were checked on
2026-10-20 against library catalogues and publisher/arXiv pages in search indexes (metadata only;
full texts not read). The others are from prior knowledge and not re-verified.

- **(V-index)** Certified reduced-basis methods, residual-based a-posteriori estimators and weak
  greedy sampling: J. S. Hesthaven, G. Rozza, B. Stamm, *Certified Reduced Basis Methods for
  Parametrized Partial Differential Equations*, SpringerBriefs in Mathematics, Springer, Cham (©2016),
  DOI 10.1007/978-3-319-22470-1. Mainly coercive problems; non-coercive ones are treated as examples.
- **(V-index)** S. Sen, K. Veroy, D. B. P. Huynh, S. Deparis, N. C. Nguyen, A. T. Patera, "Natural
  norm" a posteriori error estimators for reduced basis approximations, J. Comput. Phys. 217, 37–62
  (2006), DOI 10.1016/j.jcp.2006.02.012. Includes a non-coercive Helmholtz acoustics example with
  near-singular behaviour. The successive constraint method (D. B. P. Huynh et al., C. R. Acad. Sci.
  Paris 345, 2007) is from prior knowledge, not re-verified.
- Projection-based and variational scattering emulators in nuclear physics:
  - **(V-index)** R. J. Furnstahl, A. J. Garcia, P. J. Millican, X. Zhang, Phys. Lett. B 809, 135719
    (2020), DOI 10.1016/j.physletb.2020.135719: eigenvector continuation with the Kohn variational
    principle (not Galerkin projection).
  - **(V-index)** C. Drischler, J. A. Melendez, R. J. Furnstahl, A. J. Garcia, X. Zhang, *BUQEYE guide
    to projection-based emulators in nuclear physics*, Front. Phys. 10, 1092931 (published 17 Feb
    2023), DOI 10.3389/fphy.2022.1092931, arXiv:2212.04912.
  - From prior knowledge, not re-verified: J. A. Melendez et al., Phys. Lett. B 821, 136608 (2021);
    E. Bonilla, P. Giuliani, K. Godbey, D. Lee, Phys. Rev. C 106, 054322 (2022).
- Time–frequency concentration (why a band-limited family of oscillatory functions on an interval
  has few significant degrees of freedom): H. J. Landau, H. O. Pollak, Bell Syst. Tech. J. 41 (1962).

## 4. Published techniques vs. open questions for this project

| Published (not ours) | Question Phase 6 can test numerically |
|---|---|
| Galerkin / Petrov–Galerkin projection of scattering equations [2, 4, 6] | How many FOM solves does projection need for a 1D resonant problem, compared with the AAA rational approximation of Phases 1–5 on the same budget? |
| Residual-driven greedy selection [2], weak greedy [Hesthaven et al.] | Does the residual rank true errors near narrow resonances, where the stability constant is small? |
| POD + regression of coefficients (non-intrusive ROMs; PCA–GPR) | How much accuracy is lost when the equations are not used online? |
| Kolmogorov n-width arguments (general) | What sets the basis size in this problem: resonance width or number of resonances? |
| CC vs. CSA multi-fidelity correction (the brief's primary task) | **Blocked**: no retrievable CC/CSA data (`PHASE6_DATA_AUDIT.md`) |
