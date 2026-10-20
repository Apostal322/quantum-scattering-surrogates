# Phase 7 cost benchmark

CPU: Intel(R) Xeon(R) Processor @ 2.10GHz (1 thread). FEM N = 4331.

| stage | seconds |
|---|---|
| FEM single query | 3.59e-04 |
| FEM batched, per point | 5.80e-04 |
| ODE (DOP853, rtol 1e-10) per energy | 1.75e-02 |
| staircase S-matrix (M = 4000) per energy | 1.26e-01 |
| RB n=16: greedy offline | 7.68e-01 |
| RB n=16: single query | 3.86e-05 |
| RB n=16: batched per point | 6.09e-06 |
| RB n=16: break-even vs FEM / ODE / staircase (queries) | 2393 / 44 / 6 |
| RB n=24: greedy offline | 1.90e+00 |
| RB n=24: single query | 7.86e-05 |
| RB n=24: batched per point | 1.73e-05 |
| RB n=24: break-even vs FEM / ODE / staircase (queries) | 6776 / 109 / 15 |
| RB n=32: greedy offline | 4.85e+00 |
| RB n=32: single query | 5.09e-05 |
| RB n=32: batched per point | 2.10e-05 |
| RB n=32: break-even vs FEM / ODE / staircase (queries) | 15710 / 278 / 38 |

Note: measured while one other single-threaded Phase 7 job (the generalisation run) was active on this 4-core machine.

Staircase S-matrix batched over 1024 energies at fixed heights: 5.10e-04 s per energy (measured separately, same M).
