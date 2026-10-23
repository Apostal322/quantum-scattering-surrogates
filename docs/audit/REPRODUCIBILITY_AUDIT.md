# Reproducibility audit (audit, 2026-10-23)

**Verdict.** The project reproduces from a fresh clone:

- tests, lint, link check and quickstart all pass;
- the 17 headline numbers are recomputed exactly;
- retraining Experiment 4 with its seeds reproduces every reported aggregate;
- every Phase 7 result records its configuration, commit and a clean-tree flag, and the split
  manifest hashes match the code.

The remaining gaps: Phase 1–5 result files carry no commit hash, and absolute timings vary by
1.5–2.2× between machines.

## 1. Clean-room reproduction

Procedure, run on 2026-10-23:

1. Fresh `git clone` of `research/phase7-independent-validation` at a07e6c3 from GitHub into an
   empty directory.
2. A new `python -m venv` (Python 3.13.16), so no system packages were used. The script is
   `.../scratchpad/cleanroom/run.sh`; its commands are below.

| Step | Command | Result |
|---|---|---|
| install torch (CPU index) | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | **failed**: the index host is refused by this environment's network proxy (403) |
| install package | `pip install -e ".[dev]"` | OK; torch 2.14.1 resolved from PyPI (the fallback the README documents) |
| lint | `ruff check qscatter tests scripts` | clean |
| links | `python scripts/check_links.py` | 39 files, 0 broken |
| tests | `python -m pytest -q` | **86 passed** (119.6 s) |
| quickstart | `python -m qscatter.quickstart --fast` | OK (S-matrix resonances, AAA 1.83e-7, MLP 8.4e-2, figure written) |
| headline numbers | `python scripts/reproduce_key_results.py` | **17/17 reproduced** |
| working tree after all steps | `git status --porcelain` | clean (no step writes tracked files) |

Resolved versions: numpy 2.5.4, scipy 1.18.1, torch 2.14.1; the full list is in the clean-room
`freeze.txt`.

- **Hidden inputs.** Experiment 4/5 reference caches (`results/exp*/refs/`) are git-ignored.
  Scripts rebuild them deterministically when missing (`parametric_exp._refs`), and the clean clone
  did not have them.
- **Absolute paths.** None found: scripts use `Path(__file__).parents[...]`.

After the audit the suite has **100 tests** (86 + 14 in `tests/audit/`, about 10 s).

## 2. CI

`.github/workflows/ci.yml` runs on every push and PR:

- **test job:** ruff, link check, import smoke test, `pytest`, quickstart. It runs on Ubuntu and
  Windows with Python 3.11 and 3.12. The audit added Ubuntu 3.10 (the declared minimum) and 3.13.
- **reproduction job:** recomputes the 17 numbers and exits non-zero on mismatch.

The pip cache only stores downloads, so a missing dependency would still fail the install. No job
depends on unpublished data or credentials. Full experiments are deliberately not run in CI.

## 3. Provenance of results

| Phase | Provenance in result files | Check |
|---|---|---|
| 1–5 | config files (`results/exp*/config.json`, `systems.json`) and seeds; **no commit hash** | headline numbers recomputed (CI, clean room); Exp. 4 retrained (`results/audit_2026/exp4_strict.json`: all medians and lenient counts identical) |
| 6 | `provenance` block (config, commit, dirty flag) in each `results.json` | not rerun |
| 7 | same; all runs clean | split manifest SHA-256 reproduced from the config; core Phase 7 code unchanged between the run commit (64668e7) and the result commit (892b96b) |
| audit | `results/audit_2026/` written by `scripts/audit/*`; MLP diagnostic with provenance | — |

**Historical evidence was not overwritten.** Corrections appear as errata in documents and as new
files under `results/audit_2026/` and `results/phase7/mlp_diagnostic/`. The generated Phase 7
summary and figures were regenerated only for label changes ("certified" → "a-posteriori",
"3D" → "three parameters"). The numbers are unchanged.

## 4. Timing reproducibility

Re-measured with 15 repeats (min/median/max), one thread, OpenBLAS 0.3.34
(`scripts/audit/timing_check.py` → `results/audit_2026/timing.json`). The audit machine is a
Xeon @ 2.80 GHz, 4 logical cores, load average ≈ 1.2, because the MLP diagnostic was running.

| Quantity | Reported | Re-measured median [min–max] |
|---|---|---|
| S-matrix, 20,001 energies | 0.012 s | 0.024 s [0.017–0.028] |
| ODE solver per energy | 4.4 ms | 6.6 ms [4.1–7.6] |
| ODE scan / (30 ODE + AAA) | 670–700× | 620× |
| Phase 7 FEM single query | 3.6e-4 s | 5.9e-4 s [5.4–6.5e-4] |
| RB (n = 24) single query | 7.9e-5 s | 1.7e-4 s [1.1–2.0e-4] |
| RB (n = 24) batched, per point | 1.7e-5 s | 2.5e-5 s [2.3–2.7e-5] |
| σ_min for the a-posteriori bound, per point | 1.3 s (development) | 2.2 s |

Absolute times vary by 1.5–2.2× between virtual machines, and the within-run spread is up to 1.6×.
Ratios, and therefore conclusions such as "the fast S-matrix solver beats every surrogate" or "RB
pays off only after thousands of queries", are stable. Reports now say this; no universal speed-up
is claimed.

## 5. Not reproduced, and why

The following were not rerun:

- Experiments 2, 3, 3b and 5;
- Phase 6 `rb_1d`, `rb_3d` and `nwidth`;
- Phase 7 `validation`, `generalization`, `slice1d`, `basis` and `robustness`.

They take minutes to hours each, their code and configuration are unchanged, and their key
conclusions were checked by independent means (the physics checks C1–C8, the headline-number job,
the Experiment 4 retraining, or the Phase 7 re-tests). Their files are kept as the original
evidence.
