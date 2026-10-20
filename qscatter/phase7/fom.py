"""Phase 7 full-order model for smooth (and rectangular) potentials on a common mesh.

    -psi''(x) + V(x) psi(x) = E psi(x)      (hbar^2/2m = 1),   x in [-X, X],

with V(x) = sum_j V_j g_j(x) for fixed shape functions g_j (Gaussian bumps, or indicator functions
of rectangular barriers). Discretisation: continuous P1 finite elements on a uniform mesh; the
potential mass matrices M_j = int g_j phi_a phi_b dx are integrated with 4-point Gauss-Legendre
quadrature per element (exact for indicator functions whose edges lie on mesh nodes, O(h^8)
quadrature error for smooth g_j).

Truncation: a Gaussian never vanishes. On [-X, X] the scattering conditions are imposed as exact
transparent (Robin) conditions for V = 0 outside, so the model solves the *truncated* potential. The
truncation error is controlled by X (studied in Experiment A). Inside the domain nothing is
truncated.

The system is affine in theta(mu) with mu = (E, V_1, ..., V_J) and is compatible with
qscatter.phase6.rb.ReducedModel (same `terms`, `theta`, `f`, `observables` interface). E may be
complex (k = principal sqrt), which continues t(E) analytically to the unphysical sheet; poles of
t (resonances E_r - i Gamma/2) are found as roots of 1/t(E) (`find_pole`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix, diags

Shape = Callable[[np.ndarray], np.ndarray]

_GL_X, _GL_W = np.polynomial.legendre.leggauss(4)


def gaussian(center: float, width: float) -> Shape:
    """exp(-(x - center)^2 / width^2)."""
    return lambda x: np.exp(-((np.asarray(x) - center) ** 2) / width**2)


def indicator(lo: float, hi: float) -> Shape:
    return lambda x: ((np.asarray(x) >= lo) & (np.asarray(x) < hi)).astype(float)


def gaussian_double_barrier(VL: float, VR: float, sL: float, sR: float, d: float) -> Shape:
    """V(x) = VL exp(-(x + d/2)^2 / sL^2) + VR exp(-(x - d/2)^2 / sR^2)."""
    gL, gR = gaussian(-d / 2, sL), gaussian(d / 2, sR)
    return lambda x: VL * gL(x) + VR * gR(x)


def truncation_halfwidth(d: float, s: float, rel_cut: float) -> float:
    """X such that each Gaussian has decayed to rel_cut of its height at x = +-X."""
    return d / 2 + s * np.sqrt(np.log(1.0 / rel_cut))


@dataclass
class AffineFOM:
    """P1 FEM on a uniform mesh of [-X, X] with V = sum_j V_j g_j."""

    shapes: dict[str, Shape]  # parameter name -> shape function g_j
    X: float
    h_target: float
    breakpoints: tuple[float, ...] = ()  # nodes forced onto the mesh (rectangular edges)
    nodes: np.ndarray = field(init=False)
    terms: dict[str, csr_matrix] = field(init=False)
    f: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n_el = int(np.ceil(2 * self.X / self.h_target))
        self.nodes = np.linspace(-self.X, self.X, n_el + 1)
        h = np.diff(self.nodes)
        for b in self.breakpoints:
            if np.min(np.abs(self.nodes - b)) > 1e-9 * max(1.0, abs(b)):
                raise ValueError(f"breakpoint {b} is not a mesh node; choose X, h accordingly")
        self.x_left, self.x_right = float(self.nodes[0]), float(self.nodes[-1])
        self.length = self.x_right - self.x_left
        n = self.nodes.size

        def assemble(diag_el: np.ndarray, off_el: np.ndarray) -> csr_matrix:
            d = np.zeros(n)
            d[:-1] += diag_el
            d[1:] += diag_el
            return diags([off_el, d, off_el], [-1, 0, 1], shape=(n, n), format="csr")

        a, b = self.nodes[:-1], self.nodes[1:]
        xq = 0.5 * (a + b)[:, None] + 0.5 * h[:, None] * _GL_X[None, :]  # (n_el, 4)
        wq = 0.5 * h[:, None] * _GL_W[None, :]
        lam = (xq - a[:, None]) / h[:, None]  # local coordinate in [0, 1]
        phi0, phi1 = 1 - lam, lam
        self.terms = {
            "A0": assemble(1.0 / h, -1.0 / h),
            "M": assemble(h / 3.0, h / 6.0),
            "iB": csr_matrix(([1j, 1j], ([0, n - 1], [0, n - 1])), shape=(n, n)),
        }
        for name, g in self.shapes.items():
            gq = g(xq) * wq  # (n_el, 4): g at quadrature points times weights
            m00 = np.sum(gq * phi0 * phi0, axis=1)  # element matrix [[m00, m01], [m01, m11]]
            m01 = np.sum(gq * phi0 * phi1, axis=1)
            m11 = np.sum(gq * phi1 * phi1, axis=1)
            dg = np.zeros(n)
            dg[:-1] += m00
            dg[1:] += m11
            self.terms[f"V:{name}"] = diags([m01, dg, m01], [-1, 0, 1], shape=(n, n), format="csr")
        self.f = np.zeros(n, complex)
        self.f[0] = 1.0
        self._bands = [(T.diagonal(1), T.diagonal(0), T.diagonal(-1)) for T in self.terms.values()]

    # -------------------------------------------------------------- parameters

    @property
    def params(self) -> list[str]:
        return list(self.shapes)

    @property
    def param_names(self) -> list[str]:
        return ["E", *self.params]

    def theta(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mu = np.atleast_2d(np.asarray(mu))
        E = mu[:, 0].astype(complex)
        k = np.sqrt(E)  # principal branch: Im k < 0 for Im E < 0 (unphysical sheet)
        cols = [np.ones_like(E), -E, -k] + [mu[:, 1 + i].astype(complex) for i in range(len(self.shapes))]
        return np.stack(cols, axis=1), (-2j * k)[:, None]

    def matrix(self, mu: np.ndarray) -> csr_matrix:
        th, _ = self.theta(mu)
        return sum(c * T for c, T in zip(th[0], self.terms.values()))

    def banded(self, mu: np.ndarray) -> np.ndarray:
        th, _ = self.theta(mu)
        ab = np.zeros((3, self.nodes.size), complex)
        for c, (up, mid, lo) in zip(th[0], self._bands):
            ab[0, 1:] += c * up
            ab[1, :] += c * mid
            ab[2, :-1] += c * lo
        return ab

    # ------------------------------------------------------------------ solving

    def solve(self, mu: np.ndarray) -> np.ndarray:
        mu2 = np.atleast_2d(np.asarray(mu))
        out = np.empty((mu2.shape[0], self.nodes.size), complex)
        for i in range(mu2.shape[0]):
            _, tb = self.theta(mu2[i])
            out[i] = solve_banded((1, 1), self.banded(mu2[i]), tb[0, 0] * self.f)
        return out[0] if np.ndim(mu) == 1 else out

    def observables(self, mu: np.ndarray, psi0: np.ndarray, psiL: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.sqrt(np.atleast_2d(np.asarray(mu))[:, 0].astype(complex))
        t = psiL * np.exp(-1j * k * self.length)
        r = (psi0 - 1.0) * np.exp(2j * k * self.x_left)
        return t, r

    def amplitudes(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi = np.atleast_2d(self.solve(mu))
        return self.observables(mu, psi[:, 0], psi[:, -1])

    # ------------------------------------------------------------------ poles

    def inv_t(self, E: complex, heights: tuple[float, ...]) -> complex:
        mu = np.array([[E, *heights]], dtype=complex)
        t, _ = self.amplitudes(mu)
        return complex(1.0 / t[0])

    def find_pole(
        self, guess: complex, heights: tuple[float, ...], step_tol: float = 1e-10, maxiter: int = 60,
        depth: float = 1e-5,
    ) -> complex | None:
        """Secant iteration on 1/t(E) in the complex E plane (unphysical sheet, Im E < 0).

        Near a pole the FEM system is nearly singular, so |1/t| stalls at a round-off floor; the
        iteration therefore stops when the step is below step_tol*|z| and returns the iterate with
        the smallest |1/t|. The root is accepted only if Im z < 0 and |1/t(z)| is at least
        1/depth times smaller than |1/t| on the real axis at Re z (scale-free pole test).
        """
        z0 = complex(guess)
        z1 = z0 + 1e-6 * max(1.0, abs(z0))
        f0, f1 = self.inv_t(z0, heights), self.inv_t(z1, heights)
        best = (abs(f1), z1) if abs(f1) < abs(f0) else (abs(f0), z0)
        for _ in range(maxiter):
            if f1 == f0:
                break
            z2 = z1 - f1 * (z1 - z0) / (f1 - f0)
            z0, f0, z1 = z1, f1, z2
            f1 = self.inv_t(z1, heights)
            if abs(f1) < best[0]:
                best = (abs(f1), z1)
            if abs(z1 - z0) < step_tol * max(1.0, abs(z1)):
                break
        f_best, z = best
        if not (z.imag < 0 and z.real > 0):
            return None
        scale = abs(self.inv_t(complex(z.real, 0.0), heights))
        return z if f_best < depth * scale else None

    def sigma_min(self, mu: np.ndarray, certified: bool = False) -> float:
        """Smallest singular value of A(mu), without squaring the condition number.

        The Hermitian augmented matrix [[0, A], [A^H, 0]] has eigenvalues +-sigma_i. With the
        unknowns interleaved (x_0, y_0, x_1, y_1, ...) it is banded with half-bandwidth 3 for a
        tridiagonal A, and LAPACK's banded eigensolver returns the smallest non-negative eigenvalue
        with absolute error <= c * eps * ||A|| (backward stability + Weyl), i.e. no ||A||^2 term.

        certified=True subtracts the conservative margin 2n * eps * ||A||_1 (>= the backward-error
        bound), so the returned value is a lower bound for sigma_min in floating point; 1/sigma
        is then an upper bound for ||A^-1|| and res/sigma a certified bound for ||e||.
        """
        from scipy.linalg import eigvals_banded

        A = self.matrix(mu).tocsr()
        n = A.shape[0]
        m = 2 * n
        ab = np.zeros((4, m), complex)  # upper banded form: ab[3 - k, j] = M[j - k, j]
        rows, cols = A.nonzero()
        vals = np.asarray(A[rows, cols]).ravel()
        for i_, j_, v in zip(rows, cols, vals):
            p_, q_ = 2 * i_, 2 * j_ + 1  # M[x_i, y_j] = A[i, j]; M[y_j, x_i] = conj(A[i, j])
            lo_, hi_ = (p_, q_) if p_ < q_ else (q_, p_)
            val = v if p_ < q_ else np.conj(v)
            ab[3 - (hi_ - lo_), hi_] = val
        lam = eigvals_banded(ab, lower=False, select="i", select_range=(n, n))
        sigma = float(abs(lam[0].real))
        if certified:
            norm1 = float(abs(A).sum(axis=0).max())
            sigma = max(sigma - 2 * m * np.finfo(float).eps * norm1, 0.0)
        return sigma


def gaussian_fom(sL: float, sR: float, d: float, X: float, h: float) -> AffineFOM:
    return AffineFOM({"VL": gaussian(-d / 2, sL), "VR": gaussian(d / 2, sR)}, X=X, h_target=h)


def rectangular_fom_on_mesh(a: float, d: float, X: float, h: float) -> AffineFOM:
    """Rectangular double barrier (width a, centres +-d/2) on the same kind of mesh as the
    Gaussian model, so that both share one discrete space (needed for basis transfer)."""
    lo_L, hi_L, lo_R, hi_R = -d / 2 - a / 2, -d / 2 + a / 2, d / 2 - a / 2, d / 2 + a / 2
    return AffineFOM(
        {"VL": indicator(lo_L, hi_L), "VR": indicator(lo_R, hi_R)},
        X=X,
        h_target=h,
        breakpoints=(lo_L, hi_L, lo_R, hi_R),
    )
