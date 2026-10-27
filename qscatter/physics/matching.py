"""Third independent reference for piecewise-constant potentials: one global matching system.

The wave function is written explicitly in every region,

    x < x_0:          psi = e^{ikx} + r e^{-ikx}
    layer j:          psi = A_j e^{i q_j (x - x_j)} + B_j e^{-i q_j (x - x_{j+1})},   q_j = sqrt(E - V_j)
    x > x_N:          psi = t e^{ikx},

and continuity of psi and psi' at every interface gives 2(N+1) linear equations for the 2(N+1)
unknowns (r, A_1, B_1, ..., A_N, B_N, t), solved with dense LU. Each exponential is referenced to
the edge where it is largest, so all coefficients stay O(1) even for opaque layers (Im q_j > 0 is
chosen, so e^{i q (x - x_j)} decays to the right of x_j and e^{-i q (x - x_{j+1})} to the left of
x_{j+1}).

This shares no algebra with the Redheffer star product (`smatrix.py`), the ODE integrator or the
finite-element models; it is used by the audit (`scripts/audit/independent_physics_checks.py`) and
by tests. It is O(N^3) and meant for validation only.
"""
from __future__ import annotations

import numpy as np

from .potentials import PiecewiseConstantPotential


def matching_amplitudes(E: complex, potential: PiecewiseConstantPotential) -> tuple[complex, complex]:
    """(t, r) in the global convention of `smatrix.py`, for one (possibly complex) energy."""
    E = complex(E)
    if E.imag == 0 and E.real <= 0:
        raise ValueError("scattering requires E > 0")
    edges = np.asarray(potential.edges, dtype=float)
    V = np.asarray(potential.heights, dtype=complex)
    N = V.size
    k = np.sqrt(E)
    q = np.sqrt(E - V)
    q = np.where(q.imag < 0, -q, q)  # Im q >= 0; the solution does not depend on the sign of q
    q = np.where(np.abs(q) < 1e-300, 1e-300, q)  # q = 0 (E = V exactly) is a removable degenerate case
    n_unk = 2 * N + 2
    M = np.zeros((n_unk, n_unk), dtype=complex)
    rhs = np.zeros(n_unk, dtype=complex)
    # unknown indices: 0 -> r, 1 + 2j -> A_j, 2 + 2j -> B_j, n_unk - 1 -> t
    row = 0

    def layer_terms(j, x):
        """(coef of A_j, coef of B_j) for psi and psi' at x inside/at the edge of layer j."""
        ea = np.exp(1j * q[j] * (x - edges[j]))
        eb = np.exp(-1j * q[j] * (x - edges[j + 1]))
        return (ea, eb), (1j * q[j] * ea, -1j * q[j] * eb)

    # left edge x_0: incident + reflected = layer 0
    x0 = edges[0]
    (pa, pb), (da, db) = layer_terms(0, x0)
    M[row, 0] = np.exp(-1j * k * x0)
    M[row, 1], M[row, 2] = -pa, -pb
    rhs[row] = -np.exp(1j * k * x0)
    row += 1
    M[row, 0] = -1j * k * np.exp(-1j * k * x0)
    M[row, 1], M[row, 2] = -da, -db
    rhs[row] = -1j * k * np.exp(1j * k * x0)
    row += 1
    # interior interfaces
    for j in range(N - 1):
        x = edges[j + 1]
        (pa, pb), (da, db) = layer_terms(j, x)
        (qa, qb), (ea, eb) = layer_terms(j + 1, x)
        M[row, 1 + 2 * j], M[row, 2 + 2 * j] = pa, pb
        M[row, 3 + 2 * j], M[row, 4 + 2 * j] = -qa, -qb
        row += 1
        M[row, 1 + 2 * j], M[row, 2 + 2 * j] = da, db
        M[row, 3 + 2 * j], M[row, 4 + 2 * j] = -ea, -eb
        row += 1
    # right edge x_N: last layer = transmitted wave
    xN = edges[-1]
    (pa, pb), (da, db) = layer_terms(N - 1, xN)
    M[row, 2 * N - 1], M[row, 2 * N] = pa, pb
    M[row, n_unk - 1] = -np.exp(1j * k * xN)
    row += 1
    M[row, 2 * N - 1], M[row, 2 * N] = da, db
    M[row, n_unk - 1] = -1j * k * np.exp(1j * k * xN)
    sol = np.linalg.solve(M, rhs)
    return complex(sol[-1]), complex(sol[0])
