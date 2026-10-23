# Quantum Scattering Surrogates

**Physics-informed surrogate modeling and adaptive sampling for quantum resonance reconstruction**

[![CI](https://github.com/Apostal322/quantum-scattering-surrogates/actions/workflows/ci.yml/badge.svg)](https://github.com/Apostal322/quantum-scattering-surrogates/actions/workflows/ci.yml)

**Author:** Rostislav Pershin, undergraduate student in Computational Physics. This is coursework research code;
it has not been peer-reviewed.

The project compares four families of methods for reconstructing energy- and parameter-dependent quantum scattering
amplitudes from as few expensive solver evaluations as possible, without losing narrow resonances:

* classical interpolation and Gaussian processes;
* AAA rational approximation;
* PyTorch neural-network surrogates;
* adaptive sampling.

Every result can be regenerated from this repository.

---

**Contents**
[1 Overview](#1-research-overview-and-motivation) · [2 Problem](#2-the-mathematical-problem) ·
[3 Findings](#3-key-scientific-findings) · [4 Figures](#4-experimental-figures) · [5 Methods](#5-methods-and-models) ·
[6 PyTorch](#6-pytorch-architecture-and-training) · [7 AAA and sampling](#7-aaa-rational-approximation-and-adaptive-sampling) ·
[8 Benchmarks](#8-model-comparison-and-benchmark-results) · [9 Install](#9-installation) · [10 Quickstart](#10-quickstart) ·
[11 Reproduce](#11-reproducing-the-research) · [12 Structure](#12-project-structure) · [13 Limitations](#13-scientific-limitations) ·
[14 Coursework](#14-coursework-and-supporting-documents) · [15 Author and citation](#15-author-and-citation)

---

## 1. Research overview and motivation

Scattering observables are usually needed on dense grids of energies and system parameters, for example when
fitting potentials, propagating uncertainties or running parameter scans. Each value requires an exact
("high-fidelity") solution of the Schrödinger equation. When one solve is expensive, the obvious remedy is a
**surrogate model**: a cheap approximation built from a small number of exact evaluations.

The difficulty is **resonances**. These are narrow peaks produced by quasi-bound states. They occupy a tiny fraction
of the energy range but carry most of the physics (energy levels and lifetimes), and a surrogate that works well on
smooth curves can miss them entirely.

**Engineering problem:** reduce the number of expensive simulator calls while preserving the physically important
resonance information. Then check whether this actually saves wall-clock time.

The study answers six questions:

1. How many exact evaluations are needed?
2. How does accuracy scale with sampling density?
3. Can neural networks represent resonances?
4. When does rational approximation beat PyTorch?
5. Does adaptive sampling reduce cost?
6. When does a surrogate give a real speed-up?

## 2. The mathematical problem

The stationary Schrödinger equation in units $\hbar^2/2m = 1$ reads

$$-\psi''(x) + V(x)\,\psi(x) = E\,\psi(x), \qquad k=\sqrt{E}.$$

Here $V$ vanishes outside $[x_L, x_R]$: rectangular single, double and triple barriers, and a smooth
double-Gaussian barrier. A wave incident from the left satisfies

$$\psi = e^{ikx} + r(E)\,e^{-ikx}\ \ (x<x_L), \qquad \psi = t(E)\,e^{ikx}\ \ (x>x_R).$$

The unknowns are the complex transmission amplitude $t(E)$ and the transmission probability $T(E)=|t(E)|^2$. For real
$V$, $T+R=1$ and the $2\times2$ S-matrix is unitary. Resonances are poles of $t$ at $E_r - i\Gamma/2$, on the unphysical
sheet (lower half $k$-plane). Near such a pole

$$t(E)\approx t_{\rm bg}+\frac{c\,\Gamma/2}{E-E_r+i\Gamma/2}.$$

**Surrogate task:** given $N$ exact evaluations $\{E_i, t(E_i)\}$ (or $\{(E_i,V_0,a,d), t\}$ in the parametric
setting), predict $t$ and $T$ on a dense grid, including every resonance.

**1D transmission is not a 3D cross section.** $T(E)$ is a dimensionless probability. A 3D experiment measures
$d\sigma/d\Omega=|f(E,\theta)|^2$, with $\sigma=\frac{4\pi}{k^2}\sum_\ell(2\ell+1)\sin^2\delta_\ell$. The 1D model is
used because it is exactly verifiable and shares the pole structure. The extension to partial waves is described, not
implemented (Russian report §2.3–2.5).

## 3. Key scientific findings

These are measured findings for the 1D systems studied, not general guarantees. RMSE is the root-mean-square error
of $T(E)$ on a dense, resonance-resolved reference grid.

1. **Smooth curves need about 10 exact evaluations.** A cubic spline reaches 6·10⁻⁵ from 10 points, following the
   textbook orders N⁻², N⁻³ and N⁻⁴. A PyTorch MLP plateaus near 10⁻⁴.
2. **On resonant curves, local methods are limited by sample placement.** Splines, GP and the MLP stall at RMSE ≈
   1–2·10⁻² even with 640 uniform points. The narrowest resonance (Γ = 7.5·10⁻³) is never resolved.
3. **Rational approximation of the complex amplitude t(k) recovers every resonance from 15–40 points:** 38 + 27
   systems, widths down to 5·10⁻⁷, with asymmetric barriers, overlapping doublets, a smooth potential and noisy
   data. A window with many poles needs 80. On the reference system AAA reaches 1.8·10⁻⁷ from 20 points. The
   mechanism is that resonances are poles of the meromorphic $t(k)$. AAA is an established algorithm (Nakatsukasa,
   Sète & Trefethen, 2018). The contribution here is its systematic evaluation for this problem.
4. **The amplitude t is a better target than T = |t|²**, because T has twice as many poles.
5. **Adaptive sampling helps local interpolants only conditionally.** Splitting intervals by the change in t solves
   15 of 20 random systems, against 10 of 20 for T, but the effect depends on the phase convention.
6. **Measured failure modes.**
   - Detectability threshold: Γ/h must clearly exceed the larger of the data error and the background-approximation
     error.
   - Instability when there are too few points for the number of poles.
   - The greedy self-stopping rule stopped too early once with 10⁻⁴ noise.
7. **In the four-parameter problem (E, V₀, a, d) the MLP is the best generic method**, at 3.5·10⁻³ (the space is
   still one-dimensional). Fourier features and dissipative pretraining did not help. A **structure-exploiting
   hybrid** (single-barrier S-matrix interpolated by tensor-grid cubic splines, not a neural network, plus exact
   composition) is 2–3 orders of magnitude better; it uses the known split into two barriers and free propagation,
   which no other method is given. With the strict peak-relative criterion the MLP resolves 61–62/68 held-out
   resonances (not 66–68/68 as first reported; see `docs/audit/ISSUE_REGISTER.md`).
8. **Fewer solver calls ≠ faster.** With a fast solver no surrogate wins. With a 4.4 ms/call ODE solver, 30 points +
   AAA is **670–700× faster** than a direct 20,001-point scan (two measured runs; 620× in an audit re-run on a
   different VM, where absolute times were 1.5–2.2× slower). The MLP gains only 2.6–3.9×, because training cost
   dominates.

Negative results and corrected errors are documented: see the errata in [`docs/REPORT.md`](docs/REPORT.md), §12.2 of
the Russian report and [`CHANGELOG.md`](CHANGELOG.md).

## 4. Experimental figures

All figures are generated by `scripts/` from stored results; none is a synthetic illustration. The full gallery is
in [`docs/FIGURES.md`](docs/FIGURES.md).

**Resonant tunnelling through a symmetric double barrier** (V₀ = 10, a = 0.8, d = 2.5). These are exact solver
results. T = 1 at the three resonances (Γ = 7.5·10⁻³, 8.7·10⁻², 0.53), and T ~ 10⁻⁴ between them.

![Transmission through a double barrier](results/physics/stageB_transmission.png)

**Exact solution vs surrogates, all built from the same 160 uniform exact evaluations.** Only AAA resolves the
narrowest resonance (left panel): it infers the pole from the resonance tails.

![Reconstruction around each resonance](results/exp1/figures/exp1_stageB_zoom_N160.png)

**Error vs number of exact evaluations.** Left: smooth single barrier. Right: resonant double barrier. Solid lines
fit the complex amplitude; dashed lines fit T directly.

![Error vs number of evaluations](results/exp1/figures/exp1_rmse_uniform.png)

**Parametric surrogates (E, V₀, a, d) → t.** Median error over 40 held-out configurations (left) and 20
out-of-domain configurations (right) as a function of the simulation budget.

![Parametric surrogate comparison](results/exp4/figures/exp4_rmse_vs_budget.png)

## 5. Methods and models

| component | implementation | notes |
|---|---|---|
| Reference solver | `qscatter/physics/smatrix.py` | Per-layer transfer matrix in entire functions of $q^2=E-V$ (regular at turning points), combined with the Redheffer star product of S-matrices (stable under barriers). 0.6 µs per energy, vectorised. |
| Independent check | `qscatter/physics/ode_solver.py` | DOP853 integration (rtol 10⁻¹¹), for piecewise and smooth $V(x)$. |
| Exact poles | `qscatter/physics/poles.py` | Analytic continuation to complex E, then Newton iteration on $1/t(E)$. |
| Classical baselines | `qscatter/surrogates/baselines.py` | Linear, PCHIP, cubic spline, Gaussian process with likelihood-selected kernel. |
| Rational | `qscatter/surrogates/rational.py` | AAA (`scipy.interpolate.AAA`) of $t$ as a function of $k$, with spurious-pole filtering. |
| Neural | `qscatter/surrogates/mlp.py`, `qscatter/parametric.py` | PyTorch MLPs; Fourier features; POD+MLP; dissipative pretraining. |
| Hybrid | `qscatter/parametric.py` (`HybridSMatrix`) | Learned single-barrier $(t_b, r_b)(E,V_0,a)$ plus exact composition $t=t_bPt_b/(1-r_b^2P^2)$, $P=e^{ikd}$. |
| Adaptive sampling | `qscatter/experiments/adaptive.py`, `robustness.py` | Segment splitting by T, log T or t; GP variance; ensemble disagreement; greedy rational sampling. |

**Validation of the solver:**

| check | agreement |
|---|---|
| Analytic barrier formula | 4·10⁻¹⁵ |
| Flux conservation | 7·10⁻¹⁴ |
| Independent ODE solver | ≤ 3·10⁻¹⁰ |
| S-matrix reciprocity and unitarity | 10⁻¹⁶–10⁻¹⁴ |

## 6. PyTorch architecture and training

| model | architecture | training |
|---|---|---|
| 1D surrogate, $E \to (\mathrm{Re}\,t, \mathrm{Im}\,t)$ | MLP 1 → 64 → 64 → 2, SiLU, float64 | Full-batch Adam, lr 10⁻² with cosine decay to 10⁻⁵, up to 20k epochs. Low-data protocol: interleaved validation points, early stopping, then a refit on all points. An ablation without early stopping is in `results/exp1_mlp_no_es/`. |
| 4D surrogate, $(E,V_0,a,d) \to t$ | MLP 4 → 128 → 128 → 128 → 2, SiLU, float32 | Minibatch Adam (1024), 20k steps, cosine schedule; best checkpoint on held-out configurations. |
| Fourier-feature MLP | $[\sin 2\pi Bx, \cos 2\pi Bx]$, $B\sim\mathcal N(0,\sigma^2)$ (Tancik et al., 2020) | σ selected on validation data only. |
| POD + MLP | SVD of the training scans; MLP for the reduced coefficients | Compression error reported separately from coefficient error. |
| Dissipative pretraining | Pretrain with an absorbing well $V=-iW$, then fine-tune | Half of the budget is spent on lossy data, so the comparison is fair (idea from Nam et al.). |

Engineering details:

- Inputs are normalised with the known domain; outputs are standardised.
- Deterministic algorithms (`torch.use_deterministic_algorithms(True)`), fixed seeds, CPU.
- Hyper-parameters were chosen by validation loss, never on the test reference.
- Checkpoints carry provenance metadata: versions, git commit, data hash, seed.

**Why the MLP loses in 1D.**
- It has no built-in notion of a pole.
- Spectral bias: it failed to fit its own training points on a narrow peak, with error up to 0.68.
- The optimiser plateaus near 10⁻⁴.
- The data are noise-free, which favours interpolation.

**Why it wins among generic methods in the four-parameter problem.** Resonance ridges curve through parameter space, which defeats tensor
grids and stationary kernels.

## 7. AAA rational approximation and adaptive sampling

**AAA** (Nakatsukasa, Sète & Trefethen, 2018) represents a function in barycentric form

$$r(z)=\frac{\sum_j w_j f_j/(z-z_j)}{\sum_j w_j/(z-z_j)}.$$

It picks support points greedily and computes weights from a least-squares (SVD) problem. Our implementation choices:

- **Variable $k=\sqrt{E}$.** $t$ is meromorphic in $k$, whereas in $E$ it has a branch point at $E=0$.
- **Target: the complex amplitude, not $T$.**
- **Spurious poles removed.** Froissart doublets, i.e. negligible residue or a near-cancelling zero, are discarded.
- **Data tolerance.** The tolerance is matched to the data accuracy (rtol = 3σ for noisy data).

Diagnostics (Experiment 5):
- with N ≥ 40, all 83 resonance poles of the unseen test systems were matched to the exact poles;
- spurious and non-physical poles appear but produced no false peaks;
- sensitivity to data perturbations flags under-resolved fits.

**Greedy rational sampling.** The next sample goes where the barycentric denominator is smallest. Before each exact
evaluation the prediction there is recorded, and three consecutive look-ahead errors below 10⁻⁸ stop the loop. This
follows the greedy idea in the code of Pradovera, Hiptmair & Perugia (2025); because the output here is scalar, their
minimal rational interpolation is replaced by AAA.

**Other strategies compared:**
- segment splitting $\sqrt{\Delta\hat E^2 + \Delta y^2}$ with $y = T$, $\log T$ or $t$;
- GP posterior variance;
- MLP-ensemble disagreement.

Acquisition never sees the reference solution; a poisoned-reference test in the audit verifies this.

## 8. Model comparison and benchmark results

**Resonant double barrier, uniform sampling: RMSE of T vs number of exact evaluations N** (Experiment 1):

| method (target) | N = 20 | N = 40 | N = 160 | N = 640 |
|---|---|---|---|---|
| linear (t) | 9.2e-2 | 5.4e-2 | 3.2e-2 | 1.9e-2 |
| PCHIP (t) | 8.6e-2 | 5.2e-2 | 2.8e-2 | 1.8e-2 |
| cubic spline (T) | 9.0e-2 | 7.5e-2 | 2.5e-2 | 1.4e-2 |
| Gaussian process (T) | 8.4e-2 | 7.3e-2 | 2.5e-2 | 1.4e-2 |
| PyTorch MLP (t) | 8.3e-2 | 4.5e-2 | 2.8e-2 | — (2.5e-2 at N = 320) |
| AAA (T) | 1.6e-1 | 8.6e-9 | 8.7e-12 | 3.6e-15 |
| **AAA (t(k))** | **1.8e-7** | **3.2e-12** | **1.5e-14** | **2.4e-15** |

**Robustness on 20 random double barriers.** Success means every resonance is found and RMSE(T) ≤ 10⁻³ within at most
320 evaluations (Experiment 3):

| method | solved | median N | max N |
|---|---|---|---|
| uniform + PCHIP(t) | 0/20 | — | — |
| uniform + cubic spline(T) | 1/20 | 320 | 320 |
| adaptive segment(E,T) + PCHIP | 10/20 | 160 | 320 |
| adaptive segment(E,t) + PCHIP | 15/20 | 80 | 160 |
| uniform + AAA(T) | 20/20 | 20 | 80 |
| uniform + AAA(t) | 20/20 | 15 | 30 |
| greedy AAA | 20/20 | 15 | 20 |

**Parametric surrogate (E, V₀, a, d) → t: median test RMSE(T)** (Experiment 4; 40 held-out configurations, 3 seeds):

| method | 2,048 evals | 8,192 evals | 32,768 evals |
|---|---|---|---|
| tensor-grid cubic | 2.0e-1 | 9.8e-2 | 4.6e-2 |
| RBF (thin plate) | 1.7e-1 | 1.2e-1 | 8.0e-2 |
| Gaussian process (≤ 2,048 points) | 1.6e-1 | 2.9e-1 | — |
| POD + MLP | 1.5e-1 | 1.1e-1 | 5.0e-2 |
| MLP + Fourier features | 1.7e-1 | 4.6e-2 | 9.6e-3 |
| MLP, dissipative pretraining | 3.0e-2 | 1.1e-2 | 9.0e-3 |
| **MLP** | **9.4e-3** | **4.3e-3** | **3.5e-3** |
| **hybrid S-matrix** | **4.5e-5** | **1.3e-5** | **4.7e-6** |

For comparison, AAA run separately on each new configuration needs 16 evaluations per configuration and reaches a
median RMSE of 6.3·10⁻¹⁰. That is a different cost model: it requires computing each new configuration on demand.

**End-to-end wall-clock cost.** Target: T(E) on 20,001 energies. Single-thread CPU, run 1
(`results/cost/cost_1d.md`; run 2 in `cost_1d_run2.md`):

| solver | pipeline | exact calls | total time | RMSE(T) | speed-up vs direct scan |
|---|---|---|---|---|---|
| fast S-matrix (vectorised) | direct scan | 20,001 | 0.012 s | 0 | 1× |
| fast S-matrix | 30 points + AAA | 30 | 0.011 s | 2e-11 | ≈1× |
| fast S-matrix | 160 points + MLP | 160 | 24.5 s | 2.5e-2 | 0.0005× |
| ODE solver (4.4 ms/call) | direct scan | 20,001 | 89.5 s | 9e-12 | 1× |
| ODE solver | 30 points + AAA | 30 | 0.128 s | 5e-10 | **≈700×** (run 2: 670×) |
| ODE solver | greedy AAA (24 points) | 24 | 0.174 s | 9e-12 | ≈510× (run 2: 480×) |
| ODE solver | 160 points + MLP | 160 | 23.0 s | 2.6e-2 | 3.9× (run 2: 2.6×) |

## 9. Installation

The project needs Python ≥ 3.10; the reported results were produced with Python 3.13, and exact versions are in
`requirements-lock.txt`. Only CPU PyTorch is needed.

**Linux / macOS**

```bash
git clone https://github.com/Apostal322/quantum-scattering-surrogates.git
cd quantum-scattering-surrogates
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/Apostal322/quantum-scattering-surrogates.git
cd quantum-scattering-surrogates
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

If the CPU wheel index offers nothing for your platform (for example some macOS setups), run
`python -m pip install torch` instead. CI verifies installation, tests and the quickstart on Ubuntu and Windows with
Python 3.11 and 3.12.

## 10. Quickstart

```bash
python -m qscatter.quickstart      # ~30 s; also available as the command `qscatter-quickstart`
python -m pytest -q                # 98 tests (Phases 1–7 and the audit)
```

The quickstart:

1. solves the double barrier and lists its resonances;
2. reconstructs T(E) from 20 exact evaluations with PCHIP and with AAA;
3. trains a small PyTorch MLP;
4. writes `quickstart_output/quickstart.png`.

```text
[1] exact S-matrix solver: 99001 energies in 0.35 s
    resonance E = 0.993817, FWHM = 7.471e-03, T_peak = 1.000000
    ...
[2] N = 20 exact evaluations: RMSE(T) PCHIP = 8.63e-02, AAA(t(k)) = 1.83e-07
[3] MLP (1-64-64-2, 5000 epochs, ~20 s) on N = 40: RMSE(T) = 4.56e-02
```

As a library:

```python
import numpy as np
from qscatter.physics.potentials import double_barrier
from qscatter.physics.smatrix import compute_scattering
from qscatter.surrogates.rational import RationalAAA

pot = double_barrier(10.0, 0.8, 2.5)              # barrier height V0, barrier width a, well width d
E = np.linspace(0.1, 10.0, 20)                     # 20 exact evaluations
res = compute_scattering(E, pot)                   # res.t, res.r (complex), res.T, res.R
model = RationalAAA().fit(E, np.column_stack([res.t.real, res.t.imag]))
t_hat = model.predict(np.linspace(0.1, 10.0, 10_001)) @ np.array([1, 1j])   # reconstructed t(E)
print(model.poles())    # includes the resonance poles E_r - i*Gamma/2, e.g. 0.99380 - 0.00374j
```

Training a PyTorch surrogate:

```python
from qscatter.surrogates.mlp import MLPConfig, MLPSurrogate

mlp = MLPSurrogate((0.1, 10.0), MLPConfig(max_epochs=5000, early_stopping=False))
mlp.fit(E, np.column_stack([res.t.real, res.t.imag]))
print(mlp.predict(E[:3]))
```

## 11. Reproducing the research

Everything is deterministic: fixed seeds, deterministic PyTorch, CPU. Raw results (JSONL), configurations and
figures are written to `results/<experiment>/`. Times are approximate for a 4-core CPU.

| command | purpose | time |
|---|---|---|
| `python -m pytest -q` | 98 unit, physics, regression and audit tests | ~2–3 min |
| `python scripts/reproduce_key_results.py` | recompute 17 headline numbers and compare them with the report; exits non-zero on mismatch (also runs in CI) | ~1–3 min |
| `python scripts/audit.py` | independent audit, 10 checks: conservation, analytic/ODE agreement, leakage, counting, reproducibility | ~3–5 min |
| `python scripts/benchmark_cost.py` | wall-clock benchmark (use an idle machine) | ~2–3 min |
| `python scripts/run_exp1.py` + `analyze_exp1.py` | Experiment 1, sample efficiency (984 runs) | 1–2 h |
| `python scripts/run_exp2.py` + `analyze_exp2.py` | Experiment 2, adaptive sampling | ~30 min |
| `python scripts/run_exp3.py` + `analyze_exp3.py` | Experiment 3, robustness (38 systems) | ~20 min |
| `python scripts/run_exp3b_detectability.py` + `analyze_exp3b.py` | Experiment 3b, detectability map | ~10 min |
| `python scripts/run_exp4.py` + `analyze_exp4.py` | Experiment 4, parametric surrogates | ~1.5 h |
| `python scripts/run_exp5.py` + `analyze_exp5.py` | Experiment 5, stress test (27 unseen systems) | ~30 min |

Configurations, seeds, metrics and outputs are documented in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md); the
checklist is in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## 12. Project structure

```text
qscatter/                      Python package
  physics/                     solver layer: NumPy/SciPy only, no PyTorch
    potentials.py              single/double/asymmetric/triple barriers, smooth-potential staircase
    smatrix.py                 reference S-matrix solver, complex-energy continuation
    analytic.py, ode_solver.py textbook formulas; independent DOP853 solver
    resonances.py, poles.py    peaks and FWHM; exact complex poles
  surrogates/                  baselines.py (interpolation, GP), rational.py (AAA), mlp.py (PyTorch)
  parametric.py                four-parameter surrogates: tensor spline, RBF, GP, MLP, Fourier MLP, POD+MLP, hybrid
  experiments/                 drivers for Experiments 1–5
  phase6/                      Phase 6: FEM full-order model, reduced basis (Galerkin/LSPG), greedy, baselines
  phase7/                      Phase 7: smooth-potential FEM, poles, protocol, validation, generalisation, robustness
  data.py, metrics.py, config.py, plotting.py, quickstart.py
scripts/                       run_* / analyze_* per experiment, audit, cost benchmark, reproduction check
configs/phase6/, configs/phase7/  Phase 6 and Phase 7 experiment configurations
tests/                         pytest suite (physics, surrogates, extensions, regressions, tests/phase6, tests/phase7)
results/                       raw results, datasets, small checkpoints, figures, summary tables
docs/                          coursework report, English report, supervisor and defence material, provenance
.github/workflows/ci.yml       CI on Ubuntu + Windows; scientific reproduction job
```

## 13. Scientific limitations

* **One-dimensional models only.** Nothing is shown for 3D partial waves, Coulomb interactions, coupled channels or
  inelastic scattering.
* **AAA and rational approximation are established methods,** not contributions of this project. The project
  evaluates them systematically for scattering amplitudes and measures their limits.
* **The best method depends on the task.** AAA wins for 1D energy scans of a fixed system. The MLP is the best generic
  method for the four-parameter problem. The structure-exploiting hybrid wins when the block decomposition is known.
* **Machine learning did not beat classical methods in 1D.** This holds for the tested architectures and is not a
  general statement about neural networks.
* **Fewer solver calls do not automatically mean faster computation.** It depends on the cost per call relative to
  training and inference.
* **No general guarantees.** The detectability threshold and the self-stopping rule are empirical, and greedy
  self-certification was premature with noisy data.
* **Statistics.** 3–5 seeds per configuration. Wall-clock timings vary by 30–50 % between runs.
* **Literature.** The texts of four recent papers (2025–26) that motivated parts of the work could not be accessed
  from the development environment. Their methods were implemented from the authors' public code, and two citations
  remain unverified. See [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

## 14. Coursework and supporting documents

| document | content |
|---|---|
| [`docs/REPORT_RU.md`](docs/REPORT_RU.md) | **full coursework report (Russian):** objectives, derivations, numerical methods, ML models, all experiments, errors, limitations, references |
| [`docs/SUPERVISOR_BRIEF_RU.md`](docs/SUPERVISOR_BRIEF_RU.md) | two-page brief for the scientific supervisor (Russian) |
| [`docs/DEFENSE_PREPARATION_RU.md`](docs/DEFENSE_PREPARATION_RU.md) | 34 likely defence questions with answers (Russian), including Phase 6 |
| [`docs/KEY_FIGURES_RU.md`](docs/KEY_FIGURES_RU.md) | the eight key figures and how to explain them (Russian) |
| [`docs/REPORT.md`](docs/REPORT.md) | English research report with errata |
| [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) | experiment configurations, seeds, metrics, model architectures |
| [`docs/FIGURES.md`](docs/FIGURES.md) | figure gallery |
| [`docs/PROVENANCE.md`](docs/PROVENANCE.md) | published methods vs implementation choices vs new findings; reference-verification log |
| [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) | reproducibility checklist |
| [`results/audit/audit_report.md`](results/audit/audit_report.md) | independent audit of the results |
| [`results/reproduction_check.md`](results/reproduction_check.md) | latest recomputation of the headline numbers |
| [`CHANGELOG.md`](CHANGELOG.md) | project history, including corrected conclusions |

**Phase 6 (research branch `research/phase6-external-validation`, under review; its chapter is section 10.11 of the report).** The planned external CO2–He
multi-fidelity benchmark was blocked because its data could not be retrieved. It was replaced by a projection-based
reduced-basis experiment on the project's own validated physics.

| document | content |
|---|---|
| [`docs/PHASE6_RESULTS_RU.md`](docs/PHASE6_RESULTS_RU.md) | Phase 6 results and recommendation (Russian) |
| [`docs/PHASE6_LITERATURE_REVIEW.md`](docs/PHASE6_LITERATURE_REVIEW.md) · [RU](docs/PHASE6_LITERATURE_REVIEW_RU.md) | verified literature review, with the verification level for each source |
| [`docs/PHASE6_DATA_AUDIT.md`](docs/PHASE6_DATA_AUDIT.md) | why the external dataset could not be used |
| [`docs/PHASE6_NOVELTY_ASSESSMENT.md`](docs/PHASE6_NOVELTY_ASSESSMENT.md) | established vs. reproduced vs. new, and what still needs verification |
| [`results/phase6/SUMMARY.md`](results/phase6/SUMMARY.md) | generated tables (`scripts/phase6/analyze.py`) |

**Phase 7 (research branch `research/phase7-independent-validation`, under review; not part of the report).** Independent
validation on a smooth double-Gaussian barrier, out-of-domain and cross-family generalisation, a leakage-free comparison
of RB, POD regression, RBF, MLP, AAA and PCHIP, basis-size, certified-bound and robustness studies.

| document | content |
|---|---|
| [`docs/PHASE7_RESULTS.md`](docs/PHASE7_RESULTS.md) · [RU](docs/PHASE7_RESULTS_RU.md) | results, Phase 6 claims re-tested, coursework recommendation |
| [`docs/PHASE7_RESEARCH_PLAN.md`](docs/PHASE7_RESEARCH_PLAN.md) | questions, protocol and hypotheses fixed before evaluation |
| [`docs/PHASE7_NUMERICAL_VALIDATION.md`](docs/PHASE7_NUMERICAL_VALIDATION.md) | FEM vs ODE and exact S-matrix, convergence, resonance artefact checks |
| [`docs/PHASE7_NOVELTY_ASSESSMENT.md`](docs/PHASE7_NOVELTY_ASSESSMENT.md) · [`docs/PHASE7_LIMITATIONS.md`](docs/PHASE7_LIMITATIONS.md) | prior work, what may and may not be claimed; caveats |
| [`results/phase7/SUMMARY.md`](results/phase7/SUMMARY.md) | generated tables (`scripts/phase7/analyze.py`) |

### Skills demonstrated

**Numerical simulation (Python, NumPy, SciPy).**
- A stable S-matrix solver, validated analytically, against an independent ODE solver, and by complex-plane pole
  search.

**Surrogate modeling and PyTorch.**
- MLPs, Fourier features, POD, transfer learning, a structure-exploiting hybrid (interpolated S-matrix blocks).
- Interpolation, Gaussian processes and rational baselines, all compared under equal simulation budgets.

**Evaluation methodology.**
- Leakage tests, held-out and out-of-domain splits, bootstrap confidence intervals.
- An independent stress test on unseen systems, and error and failure-mode analysis.

**Adaptive sampling and active learning.**
- Uncertainty-, disagreement- and residual-driven acquisition.

**Performance engineering.**
- Vectorised solvers and end-to-end wall-clock benchmarks.
- Separation of sample efficiency from real speed-up.

**Software quality.**
- 98 tests, an audit script, automated reproduction of the headline numbers, ruff, CI on two operating systems, and
  checkpoint provenance.

**Resume summary.** Built a reproducible scientific-ML benchmark for reducing expensive quantum-scattering
simulations. A validated NumPy/SciPy simulator plus a comparison of interpolation, Gaussian processes, AAA rational
approximation and PyTorch surrogates under equal simulation budgets showed:

- a physics-motivated representation cut the simulations required to recover narrow resonances by 1–3 orders of
  magnitude compared with the other methods tested;
- in a four-parameter problem, a structure-exploiting hybrid (interpolated single-barrier S-matrix + exact
  composition) outperformed a generic MLP by 2–3 orders of magnitude;
- on an expensive solver the pipeline was measured end to end at 670–700× faster than a direct scan.

## 15. Author and citation

**Rostislav Pershin**, undergraduate student in Computational Physics. This repository is coursework research code,
not a peer-reviewed publication.

Please cite the software via [`CITATION.cff`](CITATION.cff) (GitHub: *Cite this repository*), and cite the original
algorithms directly, in particular AAA: Y. Nakatsukasa, O. Sète, L. N. Trefethen, *The AAA algorithm for rational
approximation*, SIAM J. Sci. Comput. 40(3), A1494–A1522 (2018).

```bibtex
@software{pershin_quantum_scattering_surrogates,
  author  = {Pershin, Rostislav},
  title   = {Quantum Scattering Surrogates: surrogate modeling and adaptive sampling
             for 1D quantum resonance reconstruction},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Apostal322/quantum-scattering-surrogates},
  note    = {Coursework research code; not peer-reviewed}
}
```


