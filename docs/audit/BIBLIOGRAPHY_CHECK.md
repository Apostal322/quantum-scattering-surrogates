# Bibliography check (audit, 2026-10-23)

**Method and limits.** Each entry of `docs/REPORT_RU.md` §14 [1]–[33] and of the Phase 7 novelty
document [34]–[40] was searched with a web-search tool. Publisher sites, arXiv, Crossref and doi.org
could not be opened from this environment (DNS failure / proxy refusal), so every status rests on
**search-index pages and excerpts**. No full text was read. "Verified" means that title, authors, venue,
volume and pages (where given) matched an index page of the publisher, arXiv, a library catalogue or
the authors' site.

| # | Entry (short) | Status | Action |
|---|---|---|---|
| 1 | Landau & Lifshitz, QM vol. III | partial: no year/edition in the entry | add "4-е изд., М.: Наука, 1989" |
| 2 | Taylor, Scattering Theory (Wiley 1972) | verified | — |
| 3 | Griffiths, Intro to QM | **correction**: "Prentice Hall / CUP" mixes the 2nd ed. (Pearson Prentice Hall 2004, Griffiths alone) and the 3rd ed. (CUP 2018, with D. F. Schroeter) | cite one edition |
| 4 | Tsu & Esaki, APL 22, 562 (1973) | verified (pp. 562–564; DOI 10.1063/1.1654509) | add DOI |
| 5 | Redheffer, J. Math. Phys. 41, 1–41 (1962) | verified (DOI 10.1002/sapm19624111) | — |
| 6 | Ko & Inkson, PRB 38, 9945 (1988) | verified (DOI 10.1103/PhysRevB.38.9945) | — |
| 7 | Hairer, Nørsett, Wanner, ODE I (2nd ed., Springer 1993) | verified | — |
| 8 | Fritsch & Carlson, SINUM 17, 238 (1980) | verified (DOI 10.1137/0717021) | — |
| 9 | de Boor, Practical Guide to Splines (1978) | verified | — |
| 10 | Rasmussen & Williams, GPML (2006) | verified | — |
| 11 | Nakatsukasa, Sète, Trefethen, AAA, SISC 40, A1494 (2018) | verified (DOI 10.1137/16M1106122) | — |
| 12 | Pradovera, Hiptmair, Perugia, J. Sci. Comput. (2025) | verified, incomplete | add vol. 104, art. 104; arXiv:2503.10194 |
| 13 | Giri et al., PRC 113, 044001 (2026) | verified | add DOI 10.1103/s6my-pqs9 |
| 14 | Nam, Park, Jang, Adv. Opt. Mater. (2026) | partial: arXiv:2601.18235 confirmed; journal listing seen; DOI seen only in a third-party list | keep marked "not confirmed" |
| 15 | Dölz & Henríquez, arXiv:2502.01859 | verified, **now published**: SIAM J. Sci. Comput. 48(2), C271–C300 (2026), DOI 10.1137/25M1731344 | cite the published version |
| 16 | Hesthaven & Ubbiali, JCP 363, 55 (2018) | verified | — |
| 17 | Berkooz, Holmes, Lumley, ARFM 25, 539 (1993) | verified | — |
| 18–22 | Adam, PyTorch, spectral bias, Fourier features, deep ensembles | verified | — |
| 23 | Sobol (1967) | verified; Russian original pp. 784–802 (English translation pp. 86–112) | keep Russian pages only |
| 24, 25 | SciPy 1.0, scikit-learn | verified | — |
| 26 | Nijholt et al., Adaptive (Zenodo 2019) | verified (DOI 10.5281/zenodo.1182437) | add DOI |
| 27 | Hesthaven, Rozza, Stamm (Springer 2016) | verified | — |
| 28 | Gnech et al., arXiv:2511.01844 | verified (no journal version found) | — |
| 29 | Catacora-Rios et al., PRC 113, 044623 (2026) | verified (DOI 10.1103/tgf9-f2st) | add DOI |
| 30 | Lei, arXiv:2512.17687 | **correction**: title ends "coupled-channels calculations"; **published** Phys. Rev. C 113, 044610 (2026), DOI 10.1103/n24x-d9gm | fix title, cite published version |
| 31 | Drischler et al., BUQEYE guide, Front. Phys. 10, 1092931 | verified | — |
| 32 | Furnstahl et al., PLB 809, 135719 (2020) | verified | — |
| 33 | Sen et al., JCP 217, 37 (2006) | verified | — |
| 34 | Maldonado, Drischler, Furnstahl, Mlinarić, PRC 112, 024002 (2025) | verified (DOI 10.1103/k77q-f82l). **Whether energy is an emulated parameter could not be determined** from abstracts | do not claim either way |
| 35 | Golub & Kahan, SIAM J. Numer. Anal. B 2, 205 (1965) | verified (DOI 10.1137/0702016) | — |
| 36 | Huynh, Rozza, Sen, Patera, C. R. Math. 345, 473 (2007) | verified | — |
| 37 | Bruschini, SPARSE, arXiv:2507.01002 | verified; **v2 removed the AAA extrapolation to complex energies** described in v1 | describe as v1 or drop the method detail |
| 38 | Nobile & Pradovera, M2AN 55, 1895 (2021) | verified | — |
| 39 | Bruno, Santana, Trefethen | **correction**: SIAM J. Sci. Comput. 48(3), A1260–A1283 (2026), DOI 10.1137/24M1690680 | fill in |
| 40 | Pradovera, SINUM 58(4) (2020) | verified, incomplete | add pp. 2265–2293, DOI 10.1137/19M1269695 |

No entry was fabricated: every work exists, and the authors are correct except the missing co-author
of the 3rd edition of [3]. The corrections above are applied in `REPORT_RU.md` §14 and
`docs/PHASE7_NOVELTY_ASSESSMENT.md` (see `ISSUE_REGISTER.md`, P2-BIB).
