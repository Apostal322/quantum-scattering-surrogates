"""Projection-based reduced-basis (RB) model and residual-driven greedy sampling.

Given an orthonormal basis V (N x r) of FOM wave-function snapshots, the reduced solution is
psi ~ V c(mu) with c obtained by *projecting the FOM equations*, not by fitting data:

- Galerkin:            V^H A(mu) V c = V^H b(mu)
- LSPG (least-squares Petrov-Galerkin): c = argmin_c || b(mu) - A(mu) V c ||_2

Because A and b are affine in theta(mu) (qscatter.phase6.fom), every reduced operator is a
theta-weighted sum of matrices precomputed once ("offline"). The online cost per parameter is
independent of the FOM size N: O(Q r^2) assembly + O(r^3) solve.

The residual norm ||b - A V c|| is also evaluated online, from a precomputed thin QR factor of the
residual space. It is
computable for *any* parameter without a FOM solve, which is what makes residual-greedy sampling
possible. Its relation to the true error is ||e|| <= ||A(mu)^{-1}|| ||res||, and ||A^{-1}|| grows
near narrow resonances, so the residual is an error *indicator*, not a guaranteed bound; Phase 6
measures how well it ranks true errors (effectivity) on held-out data.

This is different from POD *coefficient regression* (qscatter.phase6.baselines), which fits
c(mu) by interpolating training coefficients and never uses the equations online.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from qscatter.phase6.fom import FOM

Method = Literal["galerkin", "lspg"]


def orthonormalize(snapshots: np.ndarray, rtol: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """POD/SVD basis of the columns of `snapshots` (N x n); drops directions below rtol*s_max.

    Returns (V, singular values). With all snapshots kept this spans the snapshot space.
    """
    U, s, _ = np.linalg.svd(snapshots, full_matrices=False)
    keep = s > rtol * s[0]
    return U[:, keep], s


@dataclass
class ReducedModel:
    fom: FOM
    V: np.ndarray  # N x r, orthonormal columns
    offline_seconds: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        t0 = time.perf_counter()
        V = self.V
        mats = list(self.fom.terms.values())
        self.AV = [T @ V for T in mats]  # N x r each
        self.Ar = np.stack([V.conj().T @ AV for AV in self.AV])  # Q x r x r
        self.fr = V.conj().T @ self.fom.f  # r
        # LSPG / residual Gram blocks: G[q, q'] = (A_q V)^H (A_q' V),  g[q] = (A_q V)^H f
        # Residual space: res(mu) = tb f - sum_q th_q A_q V c lies in span[f, A_0 V, ..., A_Q V].
        # A thin QR of those columns gives ||res|| = ||R z(mu)|| exactly, without the round-off
        # floor (~sqrt(eps)) of Gram-matrix formulas, and a stable small least-squares problem
        # for LSPG.
        cols = np.column_stack([self.fom.f[:, None], *self.AV])
        _, self.R = np.linalg.qr(cols, mode="reduced")  # m x (1 + Q r)
        r = V.shape[1]
        self.Rf = self.R[:, 0]
        self.RA = np.stack([self.R[:, 1 + q * r : 1 + (q + 1) * r] for q in range(len(self.AV))])
        self.end0, self.endL = V[0].copy(), V[-1].copy()
        self.offline_seconds = time.perf_counter() - t0

    @property
    def r(self) -> int:
        return self.V.shape[1]

    def coefficients(self, mu: np.ndarray, method: Method = "galerkin") -> np.ndarray:
        mu = np.atleast_2d(mu)
        th, tb = self.fom.theta(mu)  # (n, Q), (n, 1)
        if method == "galerkin":
            Ar = np.einsum("nq,qij->nij", th, self.Ar)
            rhs = tb * self.fr[None, :]
            return np.linalg.solve(Ar, rhs[..., None])[..., 0]
        # LSPG: min_c || R_f tb - (sum_q th_q R_A,q) c ||, solved by batched QR (no normal equations)
        RA = np.einsum("nq,qmi->nmi", th, self.RA)
        rhs = tb * self.Rf[None, :]
        Q, Rr = np.linalg.qr(RA)
        return np.linalg.solve(Rr, np.einsum("nmi,nm->ni", Q.conj(), rhs)[..., None])[..., 0]

    def residual_norm(self, mu: np.ndarray, c: np.ndarray) -> np.ndarray:
        """||b - A V c||_2 in the small QR coordinates (no FOM-size work)."""
        mu = np.atleast_2d(mu)
        th, tb = self.fom.theta(mu)
        z = tb * self.Rf[None, :] - np.einsum("nq,qmi,ni->nm", th, self.RA, c)
        return np.linalg.norm(z, axis=1)

    def residual_norm_exact(self, mu: np.ndarray, c: np.ndarray) -> np.ndarray:
        """Same quantity assembled in FOM size (diagnostics only; avoids Gram cancellation)."""
        mu = np.atleast_2d(mu)
        th, tb = self.fom.theta(mu)
        out = np.empty(mu.shape[0])
        for i in range(mu.shape[0]):
            AVc = sum(th[i, q] * (self.AV[q] @ c[i]) for q in range(len(self.AV)))
            out[i] = np.linalg.norm(tb[i, 0] * self.fom.f - AVc)
        return out

    def amplitudes(
        self, mu: np.ndarray, method: Method = "galerkin", chunk: int = 512, residual: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, r, relative residual ||res||/||b||) for each parameter point, in chunks.

        residual=False skips the residual (NaN is returned): the online cost of a plain query.
        """
        mu = np.atleast_2d(mu)
        ts, rs, ress = [], [], []
        for i in range(0, mu.shape[0], chunk):
            m = mu[i : i + chunk]
            c = self.coefficients(m, method)
            t, r = self.fom.observables(m, c @ self.end0, c @ self.endL)
            ts.append(t)
            rs.append(r)
            ress.append(self.residual_norm(m, c) / (2 * np.sqrt(m[:, 0])) if residual  # ||b|| = 2k
                        else np.full(m.shape[0], np.nan))
        return np.concatenate(ts), np.concatenate(rs), np.concatenate(ress)

    def wavefunctions(self, mu: np.ndarray, method: Method = "galerkin") -> np.ndarray:
        return self.coefficients(np.atleast_2d(mu), method) @ self.V.T


@dataclass
class GreedyResult:
    selected: np.ndarray  # parameters chosen, in order (n x P)
    snapshots: np.ndarray  # N x n
    max_indicator: list[float]  # max relative residual over the candidate set before each add
    seconds_fom: float
    seconds_indicator: float


def residual_greedy(
    fom: FOM,
    candidates: np.ndarray,
    budget: int,
    start: np.ndarray,
    method: Method = "galerkin",
) -> GreedyResult:
    """Weak greedy: add the FOM snapshot where the reduced residual is largest.

    Uses only the residual indicator on `candidates` (no FOM solves there) and FOM solves at the
    selected points; it never sees test data.
    """
    sel = [np.asarray(start, float)]
    t0 = time.perf_counter()
    snaps = [fom.solve(sel[0])]
    t_fom = time.perf_counter() - t0
    t_ind = 0.0
    history: list[float] = []
    chosen = np.zeros(len(candidates), bool)
    while len(sel) < budget:
        V, _ = orthonormalize(np.column_stack(snaps))
        rm = ReducedModel(fom, V)
        t1 = time.perf_counter()
        _, _, res = rm.amplitudes(candidates, method)
        res[chosen] = -np.inf
        t_ind += time.perf_counter() - t1
        j = int(np.argmax(res))
        history.append(float(res[j]))
        chosen[j] = True
        sel.append(candidates[j])
        t0 = time.perf_counter()
        snaps.append(fom.solve(candidates[j]))
        t_fom += time.perf_counter() - t0
    return GreedyResult(np.array(sel), np.column_stack(snaps), history, t_fom, t_ind)
