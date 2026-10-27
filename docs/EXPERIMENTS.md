# Experiments: configuration, seeds, outputs, models

All experiment parameters are defined in code (the experiment modules listed below) and saved next to the results
(`config.json`, `systems.json`). There is no separate configuration file to keep in sync. Every experiment:

* obtains exact data only through a call-counting solver wrapper;
* evaluates only on a dense, resonance-resolved reference grid that is never used for fitting, sample selection or
  hyper-parameter choice;
* is deterministic: fixed seeds, `torch.use_deterministic_algorithms(True)`, CPU, and float64 except where noted.

## Physical systems (`qscatter/config.py`)

| name | potential | energy range | reference grid |
|---|---|---|---|
| Stage A (`single_barrier`) | rectangular barrier V₀ = 5, a = 1 | [0.1, 12] | 20,001 points |
| Stage B (`double_barrier`) | symmetric double barrier V₀ = 10, a = 0.8, d = 2.5 | [0.1, 10] | 99,001 points (ΔE = 10⁻⁴, 75 points per narrowest FWHM) |

Units: ħ²/2m = 1. The Stage B parameters were chosen by a documented scan (`scripts/stage_b_parameter_search.py`).

## Experiments

| # | module / driver | what varies | seeds | outputs |
|---|---|---|---|---|
| 1 | `qscatter/experiments/sample_efficiency.py`, `scripts/run_exp1.py` | problems A/B; N ∈ {10, 20, 40, 80, 160, 320, 640} (MLP ≤ 320); uniform / random designs; methods linear, PCHIP, cubic, GP, AAA, MLP; target t or T | random designs 0–4; MLP 0–2 | `results/exp1/` (raw JSONL, datasets, MLP checkpoints, figures, `summary.md`, `confidence_intervals.md`) |
| 1b | same, `--mlp-no-early-stopping` | MLP with fixed 20k epochs | 0–2 | `results/exp1_mlp_no_es/` |
| 2 | `qscatter/experiments/adaptive.py` | strategies: uniform, random, segment(E,T), segment(E,t), GP std, MLP ensemble; reconstruction PCHIP / cubic / GP / MLP | random designs 0–4; MLP 0–2 | `results/exp2/` |
| 3 | `qscatter/experiments/robustness.py` | 38 systems (width scan a = 0.6…2.4, 20 random double barriers [seed 12345], 4 energy intervals, 3 phase conventions, 3 noise levels) × 5 strategies × 4 reconstructions; N checkpoints 10…320 | design deterministic, noise seed 0 | `results/exp3/` |
| 3b | `scripts/run_exp3b_detectability.py` | width a = 0.6…3.2 × noise σ ∈ {0, 1e-12, 1e-9, 1e-6, 1e-4, 1e-2} × N ∈ {20, 40, 80, 160} | noise seeds 0–2 | `results/exp3b/` |
| 4 | `qscatter/parametric.py`, `qscatter/experiments/parametric_exp.py` | parametric (E, V₀, a, d) → t; budgets 2048 / 8192 / 32768; 8 methods | Sobol scramble + init seeds 0–2; test configurations seed 777; OOD configurations seed 888 | `results/exp4/` (incl. `leakage_check.md`, `query_time_aaa.json`) |
| 5 | `qscatter/experiments/stress.py` | 27 unseen systems (seed 20261013): 15 asymmetric, 4 triple-barrier doublets, 2 smooth double-Gaussian, 2 wide intervals, 4 noisy; N ∈ {20, 40, 80, 160}; 8 methods | system seed 20261013; MLP 0 | `results/exp5/` |
| cost | `scripts/benchmark_cost.py` | fast vs ODE solver; direct scan vs 4 surrogate pipelines | — | `results/cost/cost_1d.md` (run 1), `cost_1d_run2.md` (run 2) |

Reference caches (`results/exp3/refs`, `exp4/refs`, `exp5/refs`, 43 MB in total) are git-ignored. They are rebuilt
automatically and deterministically when missing.

## Metrics

* **RMSE(T):** root-mean-square error of T̂ on the reference grid. Experiments 3–5 use a trapezoid-weighted L2
  norm on a composite grid with dense windows of ±20Γ around every exact pole.
* **Resonance found:** Experiments 1–4 use T̂ > 0.5 within ±Γ of the true peak (symmetric systems). Experiment 5
  uses max|T̂ − T| ≤ 10 % of the peak within ±2Γ (valid for asymmetric peaks).
* **Spurious peak:** a peak of T̂ with prominence > 0.05, more than 3Γ from every exact pole. The first metric
  version in Experiment 4 was flawed and was withdrawn or recomputed; see `results/exp4/summary.md`.
* **Unphysical prediction:** T̂ < −10⁻⁶ or T̂ > 1 + 10⁻⁶.
* **Rational-approximant diagnostics (Experiment 5):** each pole is classified against the exact poles as matched,
  Froissart doublet, upper half-plane, or spurious near the axis. Also recorded: missed resonances, and the
  sensitivity of T̂ to a 10⁻¹⁰ perturbation of the data.

## Model architectures

| model | definition | training / fitting | notes |
|---|---|---|---|
| Linear / PCHIP / cubic spline | `qscatter/surrogates/baselines.py` (SciPy) | interpolation | cubic: not-a-knot |
| Gaussian process | scikit-learn, RBF vs Matérn-5/2 chosen by marginal likelihood | length-scale lower bound = ½ mean sample spacing (fixes a degenerate optimum) | four-parameter problem: ARD Matérn, exact GP on ≤ 2048 points |
| AAA rational | `qscatter/surrogates/rational.py` (`scipy.interpolate.AAA`, Nakatsukasa et al. 2018) | variable k = √E, target complex t; rtol 1e-13, or 3σ for noisy data | spurious poles filtered by residue |
| Greedy AAA | `qscatter/experiments/robustness.py`, `stress.py` | next point at argmin of the barycentric denominator; stop after 3 look-ahead errors < 1e-8 | idea from the gMRI code of Pradovera et al.; our adaptation |
| 1D MLP | `qscatter/surrogates/mlp.py`: 1 → 64 → 64 → 2, SiLU, float64 | Adam, lr 1e-2 with cosine decay to 1e-5, ≤ 20k epochs; interleaved validation, early stopping + refit | config chosen by validation loss (`results/mlp_config_selection.txt`) |
| four-parameter MLP | `qscatter/parametric.py`: 4 → 128 → 128 → 128 → 2, SiLU, float32 | Adam, minibatch 1024, 20k steps, best validation checkpoint | validation = held-out configurations |
| Fourier-feature MLP | same + [sin 2πBx, cos 2πBx], B ~ N(0, σ²) | σ chosen on validation data | Tancik et al. 2020 |
| POD + MLP | SVD of training scans + MLP for the coefficients | compression error reported separately | — |
| Dissipative pretraining | pretrain on an absorbing well V = −iW (W = 0.05), fine-tune lossless | half the budget lossy, same geometries | idea from Nam et al.; our 1D transfer |
| Hybrid S-matrix | single-barrier (t_b, r_b)(E, V₀, a) interpolated on a tensor grid (no neural network) + exact star-product composition | cubic interpolation | requires the block decomposition |

## Checkpoints

The Experiment 1 MLP checkpoints in `results/exp1/models/*.pt` (48 files, 1.9 MB in total) store weights,
normalisation and configuration, plus a `meta` dict with:

* library versions, git commit and creation time;
* data hash, seed, problem and target.

Load them with `MLPSurrogate.load(path)`. The metadata of checkpoints created before provenance tracking was
backfilled; the `note` field says so.
