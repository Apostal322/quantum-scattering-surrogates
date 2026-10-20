# Phase 6 external-data audit (2026-10-20)

**Verdict: the CO2–He CC/CSA dataset could not be retrieved. The multi-fidelity benchmark (brief
§4–8) was not run. Phase 6 uses the brief's §9 fallback, a projection-based reduced-basis
experiment on the project's own validated physics.**

## 1. What was attempted

| Step (brief §3) | Result |
|---|---|
| 1. Locate the supplementary files of arXiv:2610.22129 | **Failed.** `arxiv.org` and `export.arxiv.org` do not resolve (shell: HTTP 000; fetch tool: `ENOTFOUND`). Web search returns no indexed record for the identifier, so the paper's existence, title and authors are unconfirmed |
| 2. Download | Not possible (no file located) |
| 3. Provenance / integrity | Not applicable |
| 4. Licence / redistribution | Unknown |
| 5–11. Formats, units, quantum numbers, CC/CSA matching, channels, energies, precision, zeros, thresholds, transition identity | Not assessable without files |
| 12. Suitability for ML | Not assessable |

Related public sources checked:

| Source | Status |
|---|---|
| Selim, van der Avoird, Groenenboom, arXiv:2309.03781 (CO2–He, CC/CSA/VCC-IOS rates ≤ 500 K) | Indexed. Supplementary material reportedly holds figures S1–S8 and a Fortran potential routine. Machine-readable CC/CSA cross-section tables could not be confirmed, and the journal and arXiv hosts are unreachable |
| Same authors, arXiv:2206.04470 (CSA, NNCC vs CC for CO2–He) | Indexed; data not reachable |
| `github.com/ascsn/SMLR` (author code of arXiv:2510.13989) | Cloned (commit `6c1c259`). Contains QRPA nuclear-response data, **not scattering cross sections**, and **no licence file** |
| zenodo.org (common data host) | Unreachable (HTTP 000) |

## 2. What was not done, deliberately

- No cross sections were **digitised from plots** or typed in from memory.
- No synthetic "CC-like" data was generated and presented as CO2–He data.
- The CSA/CC multi-fidelity code path was **not** built, because its inputs do not exist here.
  Building it against invented data would test nothing.

## 3. How to unblock it

On a machine with normal web access:

1. Open `https://arxiv.org/abs/2610.22129` and check the title and authors. Record whether ancillary
   files (`/anc/`) or a journal supplement (`.zip`, `.txt`, `.dat`) exist, and their licence. arXiv's
   default licence covers the paper, not necessarily third-party data.
2. If they exist, save them unmodified under `data/external/co2_he/raw/` with a `SHA256SUMS` file and
   a `PROVENANCE.md` giving the URL, date, licence and file list. Then repeat items 5–12 of this audit.
3. Check arXiv:2309.03781's supplement for the same tables. It is an alternative source on the same
   collision system (CC, CSA and VCC-IOS).

Only then is the brief's multi-fidelity benchmark (direct, additive, log-ratio corrections; held-out
energies and channels; rate coefficients) meaningful.

## 4. Fallback chosen

A projection-based reduced-basis (RB) solver with residual-greedy sampling for the project's 1D
scattering problem, with the barrier heights as additional parameters (`PHASE6_RESULTS_RU.md`).
It uses no external data. Its test data are high-fidelity finite-element solutions, validated against
the exact S-matrix solver.
