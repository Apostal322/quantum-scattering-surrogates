"""High-fidelity (full-order) wave-function solver for Phase 6 reduced-basis experiments.

The 1D stationary Schrodinger equation (hbar^2/2m = 1)

    -psi''(x) + V(x) psi(x) = E psi(x),     k = sqrt(E),

is discretised with continuous piecewise-linear finite elements (P1) on [x_L, x_R], the support
of a piecewise-constant potential. Outside the support V = 0, so the scattering boundary
conditions are imposed *exactly* as Robin (transparent) conditions in local coordinates
s = x - x_L (incident amplitude 1 at s = 0):

    psi'(0)   = ik (2 - psi(0))        (incoming e^{iks} + reflected wave)
    psi'(L)   = ik psi(L)              (outgoing transmitted wave)

The weak form gives a complex-symmetric linear system with an *affine* parameter dependence

    A(mu) = A_0 - E M - k (iB) + sum_j V_j M_j,     b(mu) = -2ik f,

where M_j is the mass matrix restricted to layer j, B = e_0 e_0^T + e_N e_N^T and f = e_0.
This affine structure is what makes projection-based reduced-order models cheap online
(qscatter.phase6.rb). The global-phase observables follow the project convention
(qscatter.physics.smatrix): t = psi(L) e^{-ikL},  r = (psi(0) - 1) e^{2ik x_L}.

The FOM is validated against the exact S-matrix solver in tests/phase6 and in the Phase 6
report (P1 FEM converges as O(h^2)).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix, diags

from qscatter.physics.potentials import PiecewiseConstantPotential


@dataclass
class ParametrizedLayers:
    """Fixed layer geometry; each layer's height is a constant or a named parameter."""

    edges: tuple[float, ...]
    heights: tuple[float | str, ...]  # float = fixed height, str = parameter name
    name: str = "layers"

    @property
    def params(self) -> list[str]:
        return list(dict.fromkeys(h for h in self.heights if isinstance(h, str)))

    def potential(self, values: dict[str, float]) -> PiecewiseConstantPotential:
        hs = tuple(float(values[h]) if isinstance(h, str) else float(h) for h in self.heights)
        return PiecewiseConstantPotential(self.edges, hs, name=self.name, params=dict(values))


@dataclass
class FOM:
    """P1 finite-element full-order model with affine decomposition."""

    layers: ParametrizedLayers
    h_target: float = 2.5e-3
    nodes: np.ndarray = field(init=False)
    terms: dict[str, csr_matrix] = field(init=False)
    f: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        edges = np.asarray(self.layers.edges, float)
        pieces = []
        for lo, hi in zip(edges[:-1], edges[1:]):  # mesh aligned with every discontinuity
            n = max(2, int(np.ceil((hi - lo) / self.h_target)))
            pieces.append(np.linspace(lo, hi, n + 1)[:-1])
        self.nodes = np.concatenate([*pieces, [edges[-1]]])
        self.x_left, self.x_right = float(edges[0]), float(edges[-1])
        self.length = self.x_right - self.x_left
        h = np.diff(self.nodes)
        n = self.nodes.size
        elem_layer = np.searchsorted(edges, 0.5 * (self.nodes[:-1] + self.nodes[1:])) - 1

        def assemble(diag_el: np.ndarray, off_el: np.ndarray) -> csr_matrix:
            d = np.zeros(n)
            d[:-1] += diag_el
            d[1:] += diag_el
            return diags([off_el, d, off_el], [-1, 0, 1], shape=(n, n), format="csr")

        K = assemble(1.0 / h, -1.0 / h)
        M = assemble(h / 3.0, h / 6.0)
        B = csr_matrix(([1.0, 1.0], ([0, n - 1], [0, n - 1])), shape=(n, n))
        fixed = np.zeros_like(h)
        layer_mass: dict[str, np.ndarray] = {}
        for j, hj in enumerate(self.layers.heights):
            mask = (elem_layer == j).astype(float)
            if isinstance(hj, str):
                layer_mass[hj] = layer_mass.get(hj, np.zeros_like(h)) + mask
            else:
                fixed += hj * mask
        A0 = K + assemble(fixed * h / 3.0, fixed * h / 6.0)
        # A(mu) = sum_q theta_q(mu) * terms[q]
        self.terms = {"A0": A0.tocsr(), "M": M.tocsr(), "iB": (1j * B).tocsr()}
        for p, mask in layer_mass.items():
            self.terms[f"V:{p}"] = assemble(mask * h / 3.0, mask * h / 6.0).tocsr()
        self.f = np.zeros(n, complex)
        self.f[0] = 1.0

    # ----------------------------------------------------------------- parameters

    @property
    def param_names(self) -> list[str]:
        return ["E", *self.layers.params]

    def theta(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Affine coefficients for A (one per term, in self.terms order) and for b (on f)."""
        mu = np.atleast_2d(np.asarray(mu, float))
        E = mu[:, 0]
        k = np.sqrt(E)
        cols = [np.ones_like(E), -E, -k] + [mu[:, 1 + i] for i in range(len(self.layers.params))]
        return np.stack(cols, axis=1).astype(complex), (-2j * k)[:, None]

    def matrix(self, mu: np.ndarray) -> csr_matrix:
        th, _ = self.theta(mu)
        A = None
        for c, T in zip(th[0], self.terms.values()):
            A = c * T if A is None else A + c * T
        return A

    # ------------------------------------------------------------------- solving

    def solve(self, mu: np.ndarray) -> np.ndarray:
        """Wave function(s) at the nodes; mu has shape (P,) or (n, P). O(N) tridiagonal solve."""
        mu2 = np.atleast_2d(np.asarray(mu, float))
        th, tb = self.theta(mu2)
        out = np.empty((mu2.shape[0], self.nodes.size), complex)
        mats = list(self.terms.values())
        diags_ = [
            (m.diagonal(1), m.diagonal(0), m.diagonal(-1)) for m in mats
        ]
        for i in range(mu2.shape[0]):
            ab = np.zeros((3, self.nodes.size), complex)
            for c, (up, mid, lo) in zip(th[i], diags_):
                ab[0, 1:] += c * up
                ab[1, :] += c * mid
                ab[2, :-1] += c * lo
            out[i] = solve_banded((1, 1), ab, tb[i, 0] * self.f)
        return out[0] if np.ndim(mu) == 1 else out

    def observables(self, mu: np.ndarray, psi0: np.ndarray, psiL: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(t, r) in the project's global phase convention from psi at both ends."""
        k = np.sqrt(np.atleast_2d(np.asarray(mu, float))[:, 0])
        t = psiL * np.exp(-1j * k * self.length)
        r = (psi0 - 1.0) * np.exp(2j * k * self.x_left)
        return t, r

    def amplitudes(self, mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        psi = np.atleast_2d(self.solve(mu))
        return self.observables(mu, psi[:, 0], psi[:, -1])


def double_barrier_layers(width: float = 0.8, well: float = 2.5, symmetric: bool = False) -> ParametrizedLayers:
    """Stage-B geometry (barrier width 0.8, well 2.5) with parametric barrier heights."""
    total = 2 * width + well
    x0 = -total / 2
    edges = (x0, x0 + width, x0 + width + well, x0 + total)
    heights: tuple[float | str, ...] = ("V0", 0.0, "V0") if symmetric else ("VL", 0.0, "VR")
    return ParametrizedLayers(edges, heights, name="double_barrier_param")
