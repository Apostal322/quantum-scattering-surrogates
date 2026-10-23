"""Independent cross-check solver: direct numerical integration of the ODE.

Integrates -psi'' + V psi = E psi from the right (where psi = e^{ikx}, i.e. t = 1
before normalisation) to the left, then decomposes the solution at x_left into
incident and reflected waves. Integrating towards the left follows the physically
growing branch inside barriers, which is the stable direction for this
"shooting" problem.

This shares no algebra with smatrix.py and works for any callable V(x)
(e.g. a future smooth potential or the radial equation), so agreement between the
two methods is a meaningful validation. It is much slower, so it is used only
in tests, convergence checks and for wavefunction plots.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .potentials import PiecewiseConstantPotential


@dataclass
class ODEResult:
    E: float
    t: complex
    r: complex
    x: np.ndarray | None = None
    psi: np.ndarray | None = None


def solve_ode(
    E: float,
    potential: PiecewiseConstantPotential,
    rtol: float = 1e-11,
    atol: float = 1e-14,
    x_grid: np.ndarray | None = None,
    method: str = "DOP853",
) -> ODEResult:
    """Integrate layer by layer (exact breakpoints at the discontinuities of V)."""
    if E <= 0:
        raise ValueError("E must be positive")
    k = np.sqrt(E)
    edges = np.asarray(potential.edges, dtype=float)
    heights = np.asarray(potential.heights, dtype=float)
    xr = edges[-1]
    psi0 = np.exp(1j * k * xr)
    y = np.array([psi0.real, psi0.imag, (1j * k * psi0).real, (1j * k * psi0).imag])

    xs_out, psi_out = [], []
    for j in range(len(heights) - 1, -1, -1):
        V = heights[j]
        a, b = edges[j], edges[j + 1]

        def rhs(x, u, V=V):
            # psi'' = (V - E) psi
            return np.array([u[2], u[3], (V - E) * u[0], (V - E) * u[1]])

        t_eval = None
        if x_grid is not None:
            sel = x_grid[(x_grid >= a) & (x_grid <= b)][::-1]
            t_eval = sel if sel.size else None
        sol = solve_ivp(rhs, (b, a), y, method=method, rtol=rtol, atol=atol, t_eval=t_eval)
        if not sol.success:
            raise RuntimeError(sol.message)
        if t_eval is not None:
            xs_out.append(sol.t)
            psi_out.append(sol.y[0] + 1j * sol.y[1])
        y = sol.y[:, -1] if t_eval is None else _final_state(rhs, b, a, y, rtol, atol, method)

    psi = y[0] + 1j * y[1]
    dpsi = y[2] + 1j * y[3]
    xl = edges[0]
    A = 0.5 * (psi + dpsi / (1j * k)) * np.exp(-1j * k * xl)  # coefficient of e^{ikx}
    B = 0.5 * (psi - dpsi / (1j * k)) * np.exp(1j * k * xl)   # coefficient of e^{-ikx}
    t = 1.0 / A
    r = B / A

    x_full = psi_full = None
    if x_grid is not None:
        x_in = np.concatenate(xs_out) if xs_out else np.array([])
        p_in = np.concatenate(psi_out) / A if psi_out else np.array([], dtype=complex)
        x_full = np.asarray(x_grid, dtype=float)
        psi_full = np.empty(x_full.shape, dtype=complex)
        left = x_full < xl
        right = x_full > xr
        psi_full[left] = np.exp(1j * k * x_full[left]) + r * np.exp(-1j * k * x_full[left])
        psi_full[right] = t * np.exp(1j * k * x_full[right])
        inside = ~(left | right)
        order = np.argsort(x_in)
        x_in, p_in = x_in[order], p_in[order]
        psi_full[inside] = np.interp(x_full[inside], x_in, p_in.real) + 1j * np.interp(
            x_full[inside], x_in, p_in.imag
        )
    return ODEResult(E=E, t=t, r=r, x=x_full, psi=psi_full)


def _final_state(rhs, b, a, y, rtol, atol, method):
    sol = solve_ivp(rhs, (b, a), y, method=method, rtol=rtol, atol=atol)
    return sol.y[:, -1]


def ode_energy_scan(energies, potential, **kw):
    ts, rs = [], []
    for E in np.atleast_1d(energies):
        res = solve_ode(float(E), potential, **kw)
        ts.append(res.t)
        rs.append(res.r)
    return np.array(ts), np.array(rs)


def solve_ode_smooth(E: float, V, x_lo: float, x_hi: float, rtol: float = 1e-11, atol: float = 1e-14) -> ODEResult:
    """Direct integration for a smooth callable V(x) assumed zero outside [x_lo, x_hi].

    Used to validate the staircase (many thin layers) representation of a smooth potential, and as
    the independent reference of the Phase 7 finite-element model. Raises instead of returning an
    unconverged result.
    """
    if not E > 0:
        raise ValueError("E must be positive")
    k = np.sqrt(E)
    psi0 = np.exp(1j * k * x_hi)
    y = np.array([psi0.real, psi0.imag, (1j * k * psi0).real, (1j * k * psi0).imag])
    rhs = lambda x, u: np.array([u[2], u[3], (V(x) - E) * u[0], (V(x) - E) * u[1]])
    sol = solve_ivp(rhs, (x_hi, x_lo), y, method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(f"ODE integration failed at E = {E}: {sol.message}")
    u = sol.y[:, -1]
    psi, dpsi = u[0] + 1j * u[1], u[2] + 1j * u[3]
    A = 0.5 * (psi + dpsi / (1j * k)) * np.exp(-1j * k * x_lo)
    B = 0.5 * (psi - dpsi / (1j * k)) * np.exp(1j * k * x_lo)
    return ODEResult(E=E, t=1.0 / A, r=B / A)
