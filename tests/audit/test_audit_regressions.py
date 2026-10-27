"""Regression and independent-reference tests added by the 2026-10 audit (fast, < 1 min).

Each test targets a concrete failure mode found or checked during the audit; see
docs/audit/ISSUE_REGISTER.md for the identifiers referenced in the docstrings.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from qscatter.phase6.experiments import Scan, half_max_width, scan_metrics, spurious_peaks
from qscatter.phase6.rb import ReducedModel, orthonormalize, residual_greedy
from qscatter.phase7 import mlp
from qscatter.phase7.fom import gaussian_double_barrier, gaussian_fom, truncation_halfwidth
from qscatter.phase7.protocol import check_disjoint, make_splits, sobol
from qscatter.phase7.slice1d import secant_pole
from qscatter.physics.matching import matching_amplitudes
from qscatter.physics.ode_solver import solve_ode_smooth
from qscatter.physics.potentials import double_barrier, from_layers, single_barrier
from qscatter.physics.resonances import find_resonances, lorentzian
from qscatter.physics.smatrix import compute_scattering, transmission_complex_energy

REPO = Path(__file__).resolve().parents[2]
S, D = 0.6, 4.0
X = truncation_halfwidth(D, S, 1e-14)


def _mu(E, VL, VR):
    E = np.atleast_1d(E).astype(float)
    return np.column_stack([E, np.full(E.shape, VL), np.full(E.shape, VR)])


# ----------------------------------------------------------------- independent references


def test_global_matching_system_agrees_with_redheffer_smatrix():
    """Independent algebra (one dense interface system) vs the star product, incl. opaque and complex E."""
    rng = np.random.default_rng(7)
    for _ in range(10):
        n = int(rng.integers(1, 7))
        p = from_layers(rng.uniform(0.05, 1.5, n), rng.uniform(-5, 30, n), x0=rng.uniform(-3, 3))
        E = rng.uniform(0.01, 40, 8)
        ex = compute_scattering(E, p)
        m = np.array([matching_amplitudes(e, p) for e in E])
        np.testing.assert_allclose(m[:, 0], ex.t, atol=1e-12)
        np.testing.assert_allclose(m[:, 1], ex.r, atol=1e-12)
    opaque = single_barrier(100.0, 5.0)  # kappa a = 50
    assert matching_amplitudes(10.0, opaque)[0] == pytest.approx(compute_scattering(np.array([10.0]), opaque).t[0], rel=1e-10)
    p = double_barrier(10.0, 0.8, 2.5)
    z = 0.99 - 0.004j
    assert matching_amplitudes(z, p)[0] == pytest.approx(transmission_complex_energy(z, p)[0], rel=1e-9)


def test_weak_coupling_born_limit_including_the_phase_of_r():
    """t - 1 and r are linear in V as V -> 0 with the analytic first-Born values (global convention).

    The P1-FEM carries a free-space dispersion error, so its linear response is taken relative to
    its own V = 0 solution (audit P2-FEM-DISP).
    """
    lam, VL, VR = 1e-5, 1.0, 0.6
    E = np.array([0.5, 2.0, 4.0])
    k = np.sqrt(E)
    g = S * np.sqrt(np.pi) * np.exp(-(k * S) ** 2)
    t1 = (VL + VR) * S * np.sqrt(np.pi) / (2j * k)
    r1 = g * (VL * np.exp(-1j * k * D) + VR * np.exp(1j * k * D)) / (2j * k)
    V = gaussian_double_barrier(lam * VL, lam * VR, S, S, D)
    ode = [solve_ode_smooth(e, V, -X, X, rtol=1e-12, atol=1e-18) for e in E]
    np.testing.assert_allclose((np.array([o.t for o in ode]) - 1) / lam, t1, rtol=1e-3)
    np.testing.assert_allclose(np.array([o.r for o in ode]) / lam, r1, rtol=1e-3)
    fom = gaussian_fom(S, S, D, X, 0.0025)
    t0, r0 = fom.amplitudes(_mu(E, 0.0, 0.0))
    t, r = fom.amplitudes(_mu(E, lam * VL, lam * VR))
    np.testing.assert_allclose((t - t0) / lam, t1, rtol=1e-3)
    np.testing.assert_allclose((r - r0) / lam, r1, rtol=1e-3)


def test_fem_free_space_error_is_the_p1_dispersion_phase_error():
    """|t_h(V=0)| = 1 exactly and t_h(V=0) - 1 = k^3 h^2 L / 24 to leading order (audit P2-FEM-DISP)."""
    E = np.array([1.0, 4.0, 10.0])
    k = np.sqrt(E)
    fom = gaussian_fom(S, S, D, X, 0.005)
    t0, _ = fom.amplitudes(_mu(E, 0.0, 0.0))
    np.testing.assert_allclose(np.abs(t0), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(t0 - 1), k**3 * 0.005**2 * fom.length / 24, rtol=2e-3)


def test_translation_and_mirror_invariances_of_the_smooth_problem():
    """Shift by a: t -> t, r -> r e^{2ika}. Mirror: t unchanged (reciprocity), |r| unchanged."""
    E, a = np.array([0.5, 0.97, 3.0]), 0.37
    Vf = gaussian_double_barrier(8.0, 13.0, S, S, D)
    k = np.sqrt(E)
    sol = lambda V: [solve_ode_smooth(e, V, -X - 1, X + 1, rtol=1e-12, atol=1e-15) for e in E]  # noqa: E731
    b, s, m = sol(Vf), sol(lambda x: Vf(x - a)), sol(lambda x: Vf(-x))
    np.testing.assert_allclose([o.t for o in s], [o.t for o in b], atol=1e-10)
    np.testing.assert_allclose([o.r for o in s], np.array([o.r for o in b]) * np.exp(2j * k * a), atol=1e-10)
    np.testing.assert_allclose([o.t for o in m], [o.t for o in b], atol=1e-10)
    np.testing.assert_allclose(np.abs([o.r for o in m]), np.abs([o.r for o in b]), atol=1e-10)
    fom = gaussian_fom(S, S, D, X, 0.005)
    np.testing.assert_allclose(fom.amplitudes(_mu(E, 13.0, 8.0))[0], fom.amplitudes(_mu(E, 8.0, 13.0))[0], atol=1e-10)


def test_ode_reference_refuses_invalid_input_instead_of_returning_garbage():
    """Audit P2-ODE: solve_ode_smooth used to ignore solver failure and E <= 0."""
    V = gaussian_double_barrier(10.0, 10.0, S, S, D)
    with pytest.raises(ValueError):
        solve_ode_smooth(0.0, V, -X, X)
    with pytest.raises(RuntimeError):  # a potential that blows up makes DOP853 fail quickly
        solve_ode_smooth(1.0, lambda x: 1e300 * np.exp(np.abs(x)), -X, X)


# ----------------------------------------------------------------- error bound


def test_residual_sigma_min_bound_holds_down_to_the_round_off_floor():
    """||e||_2 <= ||res||_2 / sigma_min (exact inequality), evaluated in floating point for bases from
    under-resolved to saturated; also checks the QR-coordinate residual against the direct one."""
    f = gaussian_fom(S, S, D, X, 0.02)
    lo, hi = np.array([0.1, 8.0, 8.0]), np.array([10.0, 12.0, 12.0])
    g = residual_greedy(f, sobol(256, lo, hi, 1), 40, 0.5 * (lo + hi))
    z = f.find_pole(0.958 - 3.4e-4j, (10.0, 10.0))
    G = -2 * z.imag
    pts = np.vstack([sobol(16, lo, hi, 7), _mu(z.real + G * np.array([-1.0, 0.0, 1.0]), 10.0, 10.0)])
    psi = f.solve(pts)
    sm = np.array([f.sigma_min(m, certified=True) for m in pts])
    for n in (12, 24, 40):
        rm = ReducedModel(f, orthonormalize(g.snapshots[:, :n])[0])
        c = rm.coefficients(pts)
        ph = c @ rm.V.T
        res = rm.residual_norm(pts, c)
        direct = np.array([np.linalg.norm(f.theta(m)[1][0, 0] * f.f - f.matrix(m) @ ph[i]) for i, m in enumerate(pts)])
        # agreement up to round-off: |res_QR - res_direct| <~ eps * ||A|| * ||psi_hat|| (here ~1e-13)
        scale = np.finfo(float).eps * np.array([abs(f.matrix(m)).sum(axis=0).max() for m in pts]) * np.linalg.norm(ph, axis=1)
        assert np.all(np.abs(res - direct) <= 1e-4 * direct + 10 * scale)
        assert np.all(res / sm >= np.linalg.norm(ph - psi, axis=1))


def test_power_iteration_inverse_norm_is_only_a_lower_estimate():
    """The Phase 6 ||A^-1|| estimate never exceeds 1/sigma_min, so it cannot certify (audit P1-BOUND-P6)."""
    from qscatter.phase6.baselines import inverse_norm

    f = gaussian_fom(S, S, D, X, 0.02)
    for m in _mu(np.array([0.958, 3.5, 7.0]), 10.0, 10.0):
        assert inverse_norm(f, m) <= 1.0 / f.sigma_min(m) * (1 + 1e-10)


# ----------------------------------------------------------------- poles


def test_secant_pole_reports_none_instead_of_a_stalled_iterate():
    """Audit P2-POLE: the surrogate pole finder used to return its best iterate even without a root."""
    assert secant_pole(lambda z: 1.0 + 0j * z, 1.0 - 0.1j) is None  # 1/t never vanishes
    f = gaussian_fom(S, S, D, X, 0.01)
    z = secant_pole(lambda e: f.inv_t(e, (10.0, 10.0)), 0.958 - 2e-4j)
    assert z is not None and abs(z - f.find_pole(0.958 - 3.4e-4j, (10.0, 10.0))) < 1e-8


# ----------------------------------------------------------------- resonance metrics


def test_resonance_metrics_edge_cases():
    E = np.linspace(0.01, 2.0, 20001)
    # resonance close to E = 0: the half-maximum search must stop at the grid edge, not probe E <= 0
    calls = []
    T = lambda e: (calls.append(np.min(e)) or lorentzian(np.asarray(e), 0.012, 0.05, 1.0))  # noqa: E731
    res = find_resonances(T, E)
    assert min(calls) >= E[0]
    assert all(np.isnan(r.fwhm) or r.E_lo >= E[0] for r in res)
    # flat and small-ripple curves have no resonance
    assert find_resonances(lambda e: np.full(np.shape(e), 0.3), E) == []
    assert find_resonances(lambda e: 0.3 + 1e-3 * np.sin(200 * np.asarray(e)), E) == []
    # two overlapping Lorentzians (splitting < width) give one merged peak whose FWHM is not either width
    two = lambda e: 0.5 * (lorentzian(np.asarray(e), 1.0, 0.1, 1.0) + lorentzian(np.asarray(e), 1.03, 0.1, 1.0))  # noqa: E731
    merged = find_resonances(two, E)
    assert len(merged) == 1 and abs(merged[0].fwhm - 0.1) > 0.01
    # a peak whose half maximum is not reached before the scan edge has an undefined width (None)
    edge = lambda e: lorentzian(np.asarray(e), 1.95, 0.3, 1.0)  # noqa: E731
    assert half_max_width(edge, 1.95, 1.0, 0.3, (0.01, 2.0)) is None
    # spurious-peak count: noise bumps below prominence are ignored, a real extra peak is counted
    Ttrue = lorentzian(E, 1.0, 0.01, 1.0)
    assert spurious_peaks(E, Ttrue + 1e-3 * np.random.default_rng(0).standard_normal(E.size), [1.0 - 0.005j]) == 0
    assert spurious_peaks(E, Ttrue + lorentzian(E, 1.5, 0.01, 0.5), [1.0 - 0.005j]) == 1


def test_a_single_matching_point_does_not_count_as_a_resolved_resonance():
    """'Resolved' needs max |dT| <= 0.1 T_peak over +-2 Gamma, not one good point near the peak."""
    z = 1.0 - 0.005j
    G = 0.01
    E = np.unique(np.concatenate([np.linspace(0.5, 1.5, 2001), np.linspace(0.8, 1.2, 401)]))
    T_true = lambda e: lorentzian(np.asarray(e), 1.0, G, 1.0)  # noqa: E731
    sc = Scan({"VL": 0.0}, E, np.sqrt(T_true(E)).astype(complex), [z])
    shifted = lambda e: lorentzian(np.asarray(e), 1.0 + 0.6 * G, G, 1.0)  # noqa: E731
    m = scan_metrics(sc, np.sqrt(shifted(E)).astype(complex), shifted, T_true)
    assert not m["resonances"][0]["resolved"]
    good = scan_metrics(sc, sc.t_fom, T_true, T_true)
    assert good["resonances"][0]["resolved"]


# ----------------------------------------------------------------- ML and protocol


def test_mlp_training_is_deterministic_finite_and_fails_loudly_on_divergence():
    lo, hi = np.array([0.1, 8.0, 8.0]), np.array([10.0, 12.0, 12.0])
    Xtr, Xv = sobol(64, lo, hi, 3), sobol(32, lo, hi, 4)
    f = gaussian_fom(S, S, D, X, 0.02)
    t, tv = f.amplitudes(Xtr)[0], f.amplitudes(Xv)[0]
    p1, v1 = mlp.fit_mlp(Xtr, t, Xv, tv, lo, hi, 16, 3e-3, 200, seed=5, depth=2)
    p2, v2 = mlp.fit_mlp(Xtr, t, Xv, tv, lo, hi, 16, 3e-3, 200, seed=5, depth=2)
    assert v1 == v2 and np.array_equal(p1(Xv), p2(Xv)) and np.all(np.isfinite(p1(Xv)))
    with pytest.raises(FloatingPointError):
        mlp.fit_mlp(Xtr, t * np.nan, Xv, tv * np.nan, lo, hi, 16, 3e-3, 20, seed=0)


def test_committed_phase7_manifest_matches_the_split_code_and_sets_are_disjoint():
    """Audit of the leakage protocol: the stored SHA-256 manifest is reproducible from the config, and
    every training design (Sobol, random, MLP, greedy candidates and picks) avoids validation/test."""
    d = json.loads((REPO / "results" / "phase7" / "generalization" / "results.json").read_text())
    cfg = d["provenance"]["config"]
    sp = make_splits(cfg)
    assert sp.manifest["sha256"] == d["manifest"]["sha256"]
    lo, hi = np.array(cfg["train_lo"]), np.array(cfg["train_hi"])
    designs = [sobol(48, lo, hi, cfg["seeds"]["design_sobol"]), sobol(4096, lo, hi, cfg["seeds"]["design_mlp"]),
               sp.candidates, np.array(d["greedy"]["selected"])]
    designs += [np.random.default_rng(r).uniform(lo, hi, (48, 3)) for r in cfg["seeds"]["design_random"]]
    for Dset in designs:
        assert check_disjoint(Dset, sp.validation, sp.test, sp.test_ood)


# ----------------------------------------------------------------- reduced-basis mathematics


def test_projection_with_a_complete_basis_reproduces_the_full_order_model():
    """Galerkin and LSPG with V = an orthonormal basis of C^N must return the FOM solution exactly
    (complex conjugation / Hermitian projection, reduced right-hand side, observable extraction),
    also for complex energies used in pole searches."""
    f = gaussian_fom(S, S, D, X, 0.12)  # N ~ 100
    N = f.nodes.size
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((N, N)) + 1j * np.random.default_rng(1).standard_normal((N, N)))
    rm = ReducedModel(f, Q)
    mu = np.array([[0.7, 9.0, 11.0], [3.3, 12.0, 8.0], [6.1, 10.0, 10.0]])
    t_fom = f.amplitudes(mu)[0]
    for method in ("galerkin", "lspg"):
        np.testing.assert_allclose(rm.amplitudes(mu, method, residual=False)[0], t_fom, atol=1e-10)
    z = np.array([[0.95 - 0.01j, 10.0, 10.0]])
    np.testing.assert_allclose(rm.amplitudes(z, residual=False)[0], f.amplitudes(z)[0], atol=1e-9)


def test_pod_regression_never_uses_the_operator():
    """POD coefficient regression must be a different method from Galerkin RB: it may read the
    snapshots and extract t from wave-function values, but never assemble A(mu)."""
    from qscatter.phase6.baselines import pod_regression

    f = gaussian_fom(S, S, D, X, 0.05)
    lo, hi = np.array([0.1, 8.0, 8.0]), np.array([10.0, 12.0, 12.0])
    mu_tr, mu_te = sobol(16, lo, hi, 2), sobol(8, lo, hi, 3)
    snaps = f.solve(mu_tr).T

    class NoOperator:
        observables = f.observables

        def __getattr__(self, name):
            raise AssertionError(f"POD regression accessed the operator ({name})")

    t = pod_regression(NoOperator(), mu_tr, snaps, mu_te, lo, hi)
    assert np.all(np.isfinite(t))
    rm = ReducedModel(f, orthonormalize(snaps)[0])
    assert np.max(np.abs(t - rm.amplitudes(mu_te, residual=False)[0])) > 1e-6  # genuinely different predictions


def test_parametric_mlp_fails_loudly_when_training_diverges():
    """Audit P2-MLPDIV: MLPParam raised UnboundLocalError when no validation loss was finite."""
    from qscatter.parametric import MLPParam

    rng = np.random.default_rng(0)
    X = rng.uniform([0.2, 5.0, 0.3, 1.5], [5.0, 10.0, 0.7, 3.0], (64, 4))
    t = np.exp(1j * X[:, 0])
    with pytest.raises(FloatingPointError):
        MLPParam(hidden=(8,), steps=250, seed=0).fit(X, t * np.nan, X[:8], t[:8] * np.nan)
