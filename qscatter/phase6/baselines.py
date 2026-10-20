"""Data-driven comparators that use the SAME FOM solves as the reduced-basis model.

- POD coefficient regression: the snapshots' POD coefficients c_i = V^H psi(mu_i) are
  interpolated in mu (cubic spline in 1D, thin-plate RBF in several dimensions) and the wave
  function is reconstructed as V c(mu). The FOM equations are never used online, so nothing
  enforces the Schrodinger equation or the boundary conditions between samples.
- Direct observable interpolation: t(mu) itself is interpolated from the same samples
  (AAA rational approximation of t(k) and PCHIP in 1D; thin-plate RBF in several dimensions).

Inputs are normalised to the unit box before RBF interpolation, so no dimension dominates.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, RBFInterpolator

from qscatter.phase6.fom import FOM
from qscatter.phase6.rb import orthonormalize
from qscatter.surrogates.rational import RationalAAA


def _unit(mu: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (np.atleast_2d(mu) - lo) / (hi - lo)


def _rbf(X: np.ndarray, Y: np.ndarray, Xq: np.ndarray) -> np.ndarray:
    """Thin-plate RBF (with linear polynomial tail) for a complex (n x m) target."""
    Yri = np.column_stack([Y.real, Y.imag])
    f = RBFInterpolator(X, Yri, kernel="thin_plate_spline", degree=1)
    Z = f(Xq)
    m = Y.shape[1]
    return Z[:, :m] + 1j * Z[:, m:]


def pod_regression(
    fom: FOM,
    mu_train: np.ndarray,
    snapshots: np.ndarray,
    mu_test: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
) -> np.ndarray:
    """Transmission amplitude t from interpolated POD coefficients."""
    V, _ = orthonormalize(snapshots)
    C = (V.conj().T @ snapshots).T  # n x r
    varying = hi > lo
    if varying.sum() == 1:  # 1D parameter: spline in that coordinate
        x = mu_train[:, varying][:, 0]
        order = np.argsort(x)
        spline = CubicSpline(x[order], C[order], axis=0)
        Cq = spline(np.atleast_2d(mu_test)[:, varying][:, 0])
    else:
        Cq = _rbf(_unit(mu_train, lo, hi)[:, varying], C, _unit(mu_test, lo, hi)[:, varying])
    t, _ = fom.observables(mu_test, Cq @ V[0], Cq @ V[-1])
    return t


def direct_rbf(mu_train: np.ndarray, t_train: np.ndarray, mu_test: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    varying = hi > lo
    return _rbf(
        _unit(mu_train, lo, hi)[:, varying], t_train[:, None], _unit(mu_test, lo, hi)[:, varying]
    )[:, 0]


def direct_aaa(E_train: np.ndarray, t_train: np.ndarray, E_test: np.ndarray) -> np.ndarray:
    """AAA rational approximation of t(k), k = sqrt(E) (the Phase 1-5 method)."""
    m = RationalAAA().fit(E_train, np.column_stack([t_train.real, t_train.imag]))
    Y = m.predict(E_test)
    return Y[:, 0] + 1j * Y[:, 1]


def direct_pchip(E_train: np.ndarray, t_train: np.ndarray, E_test: np.ndarray) -> np.ndarray:
    o = np.argsort(E_train)
    re = PchipInterpolator(E_train[o], t_train[o].real)(E_test)
    im = PchipInterpolator(E_train[o], t_train[o].imag)(E_test)
    return re + 1j * im


def inverse_norm(fom: FOM, mu: np.ndarray, iters: int = 30, seed: int = 0) -> float:
    """||A(mu)^{-1}||_2 by power iteration on (A^H A)^{-1}, using two banded solves per step.

    A is complex symmetric (A^T = A), so A^H = conj(A) and solves with A^H reuse the banded form.
    """
    from scipy.linalg import solve_banded

    th, _ = fom.theta(np.atleast_2d(mu))
    ab = np.zeros((3, fom.nodes.size), complex)
    for c, T in zip(th[0], fom.terms.values()):
        ab[0, 1:] += c * T.diagonal(1)
        ab[1, :] += c * T.diagonal(0)
        ab[2, :-1] += c * T.diagonal(-1)
    abH = np.conj(ab)
    x = np.random.default_rng(seed).standard_normal(fom.nodes.size) + 0j
    x /= np.linalg.norm(x)
    est = 0.0
    for _ in range(iters):
        y = solve_banded((1, 1), abH, x)  # A^{-H} x
        z = solve_banded((1, 1), ab, y)  # A^{-1} A^{-H} x
        est = float(np.sqrt(np.linalg.norm(z)))
        x = z / np.linalg.norm(z)
    return est
