# Phase 6 cost benchmark (scripts/phase6/benchmark_cost.py)

Intel(R) Xeon(R) Processor @ 2.10GHz; Linux-6.18.44-fc-v139-x86_64-with-glibc2.39; Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1; OMP_NUM_THREADS=1. FOM 1641 nodes; RB from 24 greedy FOM solves, basis size 24. Medians of 9 repetitions. On the shared cloud VM used for the reported numbers, repeated runs differed by up to ~1.5x; compare orders of magnitude, not the last digit.

| stage | time |
|---|---|
| FOM solve, single query | 372.9 µs |
| exact S-matrix solver, single query | 164.1 µs |
| exact S-matrix solver, batched (per energy) | 0.7 µs |
| offline: 24 snapshots | 3.6 ms |
| offline: SVD basis | 1.7 ms |
| offline: reduced operators + residual QR | 48.2 ms |
| offline: residual greedy to n = 24 (incl. its FOM solves) | 1300.2 ms |
| offline total, without / with greedy | 53.5 ms / 1350.2 ms |
| RB single query (reduced solve + t) | 64.1 µs |
| RB single query, reduced solve only | 44.3 µs |
| RB batched, per query (without / with residual) | 14.1 µs / 51.9 µs |
| RB batched: observable reconstruction / residual norm, per query | 0.1 µs / 56.0 µs |

| break-even number of queries | value |
|---|---|
| vs FOM, offline without greedy | 174 |
| vs FOM, offline with greedy | 4373 |
| vs exact S-matrix solver | 536 |
