"""Phase 7: smooth-potential FEM, poles, protocol, certified residual bound, runners (fast)."""
import json
from pathlib import Path

import numpy as np
import pytest

from qscatter.phase6.rb import ReducedModel, orthonormalize
from qscatter.phase7 import experiments as ex
from qscatter.phase7.fom import (
    gaussian,
    gaussian_double_barrier,
    gaussian_fom,
    rectangular_fom_on_mesh,
    truncation_halfwidth,
)
from qscatter.phase7.protocol import check_disjoint, make_splits
from qscatter.phase7.resonances import find_poles
from qscatter.physics.ode_solver import solve_ode_smooth
from qscatter.physics.potentials import double_barrier
from qscatter.physics.smatrix import compute_scattering

REPO = Path(__file__).resolve().parents[2]
S, D = 0.6, 4.0


def _mu(E, VL, VR):
    E = np.atleast_1d(E)
    return np.column_stack([E, np.full_like(E, VL), np.full_like(E, VR)])


def test_gaussian_potential_definition_and_asymptotics():
    V = gaussian_double_barrier(10.0, 12.0, S, 0.5, D)
    assert V(-D / 2) == pytest.approx(10.0 + 12.0 * np.exp(-(D**2) / 0.25), rel=1e-12)
    assert V(D / 2) == pytest.approx(12.0, rel=1e-6)
    X = truncation_halfwidth(D, S, 1e-14)
    assert gaussian(D / 2, S)(X) == pytest.approx(1e-14, rel=1e-9)  # decayed to rel_cut at x = X
    assert V(np.array([-50.0, 50.0])).max() < 1e-300


def test_fom_matches_exact_smatrix_for_rectangular_barriers_on_the_shared_mesh():
    E = np.linspace(0.3, 9.5, 25)
    ex_ = compute_scattering(E, double_barrier(10.0, 1.0, D - 1.0, height_right=12.0))
    errs = []
    for h in (0.0125, 0.00625):  # both keep the barrier edges on mesh nodes for X = 5.4125
        t, r = rectangular_fom_on_mesh(1.0, D, 5.4125, h).amplitudes(_mu(E, 10.0, 12.0))
        errs.append(max(np.max(np.abs(t - ex_.t)), np.max(np.abs(r - ex_.r))))
    assert 3.5 < errs[0] / errs[1] < 4.5  # second order towards the exact solution
    with pytest.raises(ValueError):
        rectangular_fom_on_mesh(1.0, D, 5.401, 0.0025)  # edges off the mesh are refused


def test_fom_converges_to_the_independent_ode_solution_at_second_order():
    X = truncation_halfwidth(D, S, 1e-12)
    V = gaussian_double_barrier(10.0, 11.0, S, S, D)
    E = np.linspace(0.5, 9.0, 12)
    t_ode = np.array([solve_ode_smooth(e, V, -X, X, rtol=1e-12, atol=1e-15).t for e in E])
    errs = []
    for h in (0.01, 0.005):
        t, r = gaussian_fom(S, S, D, X, h).amplitudes(_mu(E, 10.0, 11.0))
        errs.append(np.max(np.abs(t - t_ode)))
        assert np.max(np.abs(np.abs(t) ** 2 + np.abs(r) ** 2 - 1)) < 1e-10  # flux conservation
    assert 3.5 < errs[0] / errs[1] < 4.5


def test_complex_energy_continuation_and_pole_finding():
    g = gaussian_fom(S, S, D, truncation_halfwidth(D, S, 1e-12), 0.005)
    E = np.linspace(0.5, 3, 7)
    t_real = g.amplitudes(_mu(E, 10.0, 10.0))[0]
    t_cplx = g.amplitudes(np.column_stack([E.astype(complex), np.full(7, 10.0), np.full(7, 10.0)]))[0]
    np.testing.assert_allclose(t_cplx, t_real, atol=1e-13)
    poles = find_poles(g, (10.0, 10.0), (0.1, 5.0))
    assert len(poles) == 2 and all(p.imag < 0 for p in poles)
    assert poles[0].real == pytest.approx(0.95846, abs=1e-4)  # narrow resonance (Gamma ~ 7e-4)
    for p in poles:
        assert abs(g.inv_t(p, (10.0, 10.0))) < 1e-5 * abs(g.inv_t(complex(p.real), (10.0, 10.0)))


def test_sigma_min_matches_dense_svd_and_certifies_the_residual_bound():
    g = gaussian_fom(S, S, D, truncation_halfwidth(D, S, 1e-8), 0.05)
    mu = _mu(0.958, 10.0, 10.0)[0]
    s_dense = np.linalg.svd(g.matrix(mu).toarray(), compute_uv=False)[-1]
    assert g.sigma_min(mu) == pytest.approx(s_dense, rel=1e-8)
    assert g.sigma_min(mu, certified=True) <= s_dense  # certified value is a lower bound
    S_ = g.solve(_mu(np.linspace(0.5, 9.0, 5), 10.0, 10.0)).T
    rm = ReducedModel(g, orthonormalize(S_)[0])
    for m in _mu(np.array([0.958, 3.4, 7.7]), 9.0, 11.0):
        c = rm.coefficients(m[None])
        err = np.linalg.norm(c[0] @ rm.V.T - g.solve(m))
        assert err <= rm.residual_norm(m[None], c)[0] / g.sigma_min(m, certified=True)


def test_reduced_model_reproduces_gaussian_snapshots():
    g = gaussian_fom(S, S, D, truncation_halfwidth(D, S, 1e-10), 0.01)
    mu = np.array([[1.0, 9.0, 11.0], [4.0, 10.0, 10.0], [7.5, 12.0, 8.0]])
    rm = ReducedModel(g, orthonormalize(g.solve(mu).T)[0])
    np.testing.assert_allclose(rm.amplitudes(mu)[0], g.amplitudes(mu)[0], atol=1e-10)


def _small_cfg():
    cfg = json.loads((REPO / "configs" / "phase7" / "generalization.json").read_text())
    cfg.update(h=0.0125, n_candidates=128, n_validation=32, n_test=64, n_test_ood_per_box=16,
               n_scans_test=1, n_scans_ood_per_box=1, budgets=[8, 12], mlp_budgets=[32],
               mlp_grid={"width": [16], "lr": [0.003], "epochs": 50},
               scan={"n_uniform": 801, "n_window": 41}, rect_budgets=[8], rect_enrich_base=8,
               enrich_budgets=[2], n_effectivity=8, effectivity_budgets=[8])
    return cfg


def test_protocol_splits_are_disjoint_frozen_and_hashed():
    cfg = _small_cfg()
    a, b = make_splits(cfg), make_splits(cfg)
    assert a.manifest["sha256"] == b.manifest["sha256"]  # reproducible
    assert check_disjoint(a.candidates, a.validation, a.test, a.test_ood)
    lo, hi = np.array(cfg["train_lo"]), np.array(cfg["train_hi"])
    assert np.all((a.test >= lo) & (a.test <= hi))
    assert np.all(np.any((a.test_ood[:, 1:] < lo[1:]) | (a.test_ood[:, 1:] > hi[1:]), axis=1))
    assert cfg["baseline_design"] in ("sobol",)  # declared before evaluation


def test_generalization_runner_end_to_end_on_a_small_config():
    d = ex.run_generalization(_small_cfg())
    designs = {r["design"] for r in d["rows"]}
    assert {"greedy", "sobol"} <= designs
    sob = [r for r in d["rows"] if r["design"] == "sobol"]
    assert all("rbf_t" in r and "pod_regression" in r for r in sob)
    assert not any("rbf_t" in r for r in d["rows"] if r["design"] != "sobol")  # no best-of-design
    assert d["rbf_kernel"] in d["rbf_kernel_validation"]
    e = d["effectivity"][0]
    assert e["random"]["certified_bound_holds_psi"] == 1.0


def _gen_coarse():
    gen = json.loads((REPO / "configs" / "phase7" / "generalization.json").read_text())
    gen.update(h=0.0125, n_scans_test=1)
    return gen


def test_slice1d_runner_on_a_small_config():
    from qscatter.phase7.slice1d import aaa_adaptive_design, run_slice1d

    g = gaussian_fom(S, S, D, 5.4125, 0.0125)
    t_of = lambda E: g.amplitudes(_mu(E, 10.0, 10.0))[0]  # noqa: E731
    E, _ = aaa_adaptive_design(t_of, np.linspace(0.1, 10, 128), 12)
    assert len(np.unique(E)) == 12  # no energy is sampled twice
    d = run_slice1d({"splits": _gen_coarse(), "n_energy_candidates": 128, "budgets": [24],
                     "scan": {"n_uniform": 401, "n_window": 21}})
    row = d["scans"][0]["rows"][0]
    assert row["rb_greedy"]["max_abs_dt"] < 1e-3 and all(e["matched"] for e in row["rb_greedy"]["pole_errors"])


def test_basis_study_counts_resonances_and_basis_size():
    from qscatter.phase7.basis import study

    cfg = json.loads((REPO / "configs" / "phase7" / "basis.json").read_text())
    cfg.update(h=0.0125, n_candidates=128, n_validation=50, n_dense=200, n_max=20)
    r = study(0.6, 4.0, 10.0, 5.0, cfg)
    assert r["n_resonances"] == 2
    assert r["n_for_accuracy"]["0.0001"] is not None and r["n_for_accuracy"]["0.0001"] <= 20
    assert np.all(np.diff(r["singular_values"]) <= 1e-12)


def test_robustness_runner_on_a_small_config():
    from qscatter.phase7.robustness import run_robustness

    cfg = json.loads((REPO / "configs" / "phase7" / "robustness.json").read_text())
    cfg.update(splits=_gen_coarse(), n_lines=1, n_energy_candidates=128, budget=16, seeds=[1], noise_levels=[1e-8],
               coarse_meshes=[0.025], jitter=[0.3], scan={"n_uniform": 401, "n_window": 21})
    cfg["triple"] = dict(cfg["triple"], middle_heights=[16.0], budgets=[16])
    d = run_robustness(cfg)
    line = d["lines"][0]
    assert line["precision"]["complex64"] > line["precision"]["complex128"]
    assert {"aaa_uniform", "aaa_adaptive", "pchip_uniform", "rb_greedy_noisy_snapshots"} <= set(line["noise"][0])
    assert d["doublets"]["configs"][0]["adjacent_pairs"]
