"""Phase 6: full-order FEM model, projection-based reduced basis, greedy sampling, baselines."""
import numpy as np
import pytest

from qscatter.phase6 import baselines as bl
from qscatter.phase6.fom import FOM, double_barrier_layers
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy
from qscatter.physics.smatrix import compute_scattering

COARSE = 0.01


@pytest.fixture(scope="module")
def fom3() -> FOM:
    return FOM(double_barrier_layers(), COARSE)


def _mu3(E, VL=10.0, VR=12.0):
    E = np.atleast_1d(E)
    return np.column_stack([E, np.full_like(E, VL), np.full_like(E, VR)])


def test_fom_converges_at_second_order_to_the_exact_solver_with_the_same_phase():
    lay = double_barrier_layers()
    E = np.linspace(0.2, 9.5, 40)
    exact = compute_scattering(E, lay.potential({"VL": 10.0, "VR": 12.0}))
    errs = []
    for h in (0.02, 0.01):
        t, r = FOM(lay, h).amplitudes(_mu3(E))
        errs.append(np.max(np.abs(t - exact.t)))  # complex: magnitude and global phase
        assert np.max(np.abs(np.abs(t) ** 2 + np.abs(r) ** 2 - 1)) < 1e-10  # flux conserved
    assert 3.5 < errs[0] / errs[1] < 4.5  # O(h^2)


def test_affine_terms_reassemble_the_operator(fom3: FOM):
    mu = _mu3(3.3)[0]
    psi = fom3.solve(mu)
    _, tb = fom3.theta(mu)
    res = fom3.matrix(mu) @ psi - tb[0, 0] * fom3.f
    assert np.linalg.norm(res) < 1e-10 * np.linalg.norm(tb[0, 0] * fom3.f)


def test_reduced_model_reproduces_its_training_snapshots(fom3: FOM):
    mu = _mu3(np.array([0.7, 2.0, 5.5, 8.0]), VL=9.0, VR=13.0)
    S = fom3.solve(mu).T
    rm = ReducedModel(fom3, orthonormalize(S)[0])
    t_fom, _ = fom3.amplitudes(mu)
    for method in ("galerkin", "lspg"):
        t, _, res = rm.amplitudes(mu, method)
        assert np.max(np.abs(t - t_fom)) < 1e-9
        assert np.max(res) < 1e-9


def test_fast_residual_matches_the_full_size_residual_and_lspg_is_optimal(fom3: FOM):
    S = fom3.solve(_mu3(np.linspace(0.5, 9.5, 6))).T
    rm = ReducedModel(fom3, orthonormalize(S)[0])
    mu = _mu3(np.array([1.3, 4.4, 7.1]), VL=8.0, VR=11.0)
    cg, cl = rm.coefficients(mu, "galerkin"), rm.coefficients(mu, "lspg")
    fast, full = rm.residual_norm(mu, cg), rm.residual_norm_exact(mu, cg)
    np.testing.assert_allclose(fast, full, rtol=1e-6)
    assert np.all(rm.residual_norm(mu, cl) <= fast * (1 + 1e-10))  # LSPG minimises the residual


def test_residual_times_inverse_norm_bounds_the_true_error(fom3: FOM):
    S = fom3.solve(_mu3(np.linspace(0.5, 9.5, 5))).T
    rm = ReducedModel(fom3, orthonormalize(S)[0])
    for mu in _mu3(np.array([0.9, 3.1, 6.6])):
        c = rm.coefficients(mu)
        err = np.linalg.norm(c[0] @ rm.V.T - fom3.solve(mu))
        bound = bl.inverse_norm(fom3, mu) * rm.residual_norm(mu[None], c)[0]
        assert err <= bound * (1 + 1e-8)


def test_inverse_norm_matches_dense_svd():
    small = FOM(double_barrier_layers(), 0.05)
    mu = _mu3(1.1)[0]
    s = np.linalg.svd(small.matrix(mu).toarray(), compute_uv=False)
    assert bl.inverse_norm(small, mu, iters=80) == pytest.approx(1 / s[-1], rel=1e-8)


def test_greedy_is_deterministic_uses_candidates_and_never_repeats(fom3: FOM):
    rng = np.random.default_rng(0)
    cand = np.column_stack([rng.uniform(0.1, 10, 200), rng.uniform(6, 14, (200, 2))])
    a = residual_greedy(fom3, cand, 8, np.array([5.0, 10.0, 10.0]))
    b = residual_greedy(fom3, cand, 8, np.array([5.0, 10.0, 10.0]))
    np.testing.assert_array_equal(a.selected, b.selected)
    picked = a.selected[1:]
    assert len({tuple(p) for p in picked}) == len(picked)
    assert all(any(np.array_equal(p, c) for c in cand) for p in picked)
    assert a.snapshots.shape == (fom3.nodes.size, 8)


def test_greedy_indicator_decreases_overall(fom3: FOM):
    cand = _mu3(np.linspace(0.15, 9.9, 300), VL=10.0, VR=10.0)
    g = residual_greedy(fom3, cand, 14, np.array([5.0, 10.0, 10.0]))
    assert g.max_indicator[-1] < 1e-3 * g.max_indicator[0]


def test_pod_regression_and_direct_interpolation_reproduce_training_values(fom3: FOM):
    lo, hi = np.array([0.1, 6, 6]), np.array([10, 14, 14])
    rng = np.random.default_rng(3)
    mu = np.column_stack([rng.uniform(0.1, 10, 12), rng.uniform(6, 14, (12, 2))])
    S = fom3.solve(mu).T
    t, _ = fom3.observables(mu, S[0], S[-1])
    np.testing.assert_allclose(bl.pod_regression(fom3, mu, S, mu, lo, hi), t, atol=1e-8)
    np.testing.assert_allclose(bl.direct_rbf(mu, t, mu, lo, hi), t, atol=1e-8)
