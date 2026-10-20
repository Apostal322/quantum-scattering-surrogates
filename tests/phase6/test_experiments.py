"""Phase 6 experiment helpers: test-scan construction, peak metrics, AAA greedy design, configs."""
import json
from pathlib import Path

import numpy as np
import pytest

from qscatter.phase6 import experiments as ex
from qscatter.phase6.fom import FOM, double_barrier_layers

REPO = Path(__file__).resolve().parents[2]


def test_scan_resolves_every_pole_and_scores_the_truth_as_perfect():
    fom = FOM(double_barrier_layers(symmetric=True), 0.01)
    scan = ex.make_scan(fom, {"V0": 10.0}, (0.1, 10.0), 2001, 201)
    assert len(scan.poles) == 3
    for z in scan.poles:  # dense window around each pole
        G = -2 * z.imag
        assert np.sum(np.abs(scan.E - z.real) < G) > 10

    def T_fom(e):
        return np.abs(fom.amplitudes(np.column_stack([e, np.full_like(e, 10.0)]))[0]) ** 2

    m = ex.scan_metrics(scan, scan.t_fom, T_fom, T_fom)
    assert m["max_abs_dT"] == 0.0
    assert all(r["resolved"] and r["peak_shift_over_gamma"] < 1e-6 for r in m["resonances"])


def test_aaa_greedy_design_is_deterministic_and_uses_candidates():
    fom = FOM(double_barrier_layers(symmetric=True), 0.02)
    cand = np.linspace(0.1, 10, 500)
    a = ex.aaa_greedy_design(fom, cand, 10.0, 9)
    b = ex.aaa_greedy_design(fom, cand, 10.0, 9)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (9, 2) and np.all(np.isin(a[4:, 0], cand))
    assert len(np.unique(a[:, 0])) == 9


def test_configs_keep_train_candidate_and_test_sources_separate():
    c3 = json.loads((REPO / "configs" / "phase6" / "rb_3d.json").read_text())
    seeds = {c3["seed_candidates"], c3["seed_test"], c3["seed_scans"], c3["seed_sobol_design"]}
    assert len(seeds) == 4 and not seeds & set(c3["random_seeds"])
    c1 = json.loads((REPO / "configs" / "phase6" / "rb_1d.json").read_text())
    assert c1["scan_uniform"] > 4 * c1["n_candidates"]  # test grid much denser than candidates


def test_provenance_records_config_commit_and_dirty_flag():
    p = ex.provenance({"x": 1})
    assert p["config"] == {"x": 1} and "git_commit" in p and "code_modified_since_commit" in p


def test_half_max_width_matches_a_lorentzian_and_handles_undefined_widths():
    E0, G = 2.0, 0.01

    def lor(e):
        return 1.0 / (1.0 + ((np.asarray(e) - E0) / (G / 2)) ** 2)

    assert ex.half_max_width(lor, E0, 1.0, G, (0.1, 10.0)) == pytest.approx(G, rel=1e-9)

    def flat(e):  # peak on a background that never drops to half maximum
        return 0.8 + 0.2 * lor(e)

    assert ex.half_max_width(flat, E0, 1.0, G, (0.1, 10.0)) is None


def test_spurious_peak_count_ignores_true_resonances_and_flags_extra_peaks():
    E = np.linspace(0.0, 4.0, 4001)
    pole = 1.0 - 0.005j  # Gamma = 0.01

    def bump(c, w):
        return np.exp(-(((E - c) / w) ** 2))

    T_true = bump(1.0, 0.01)
    assert ex.spurious_peaks(E, T_true, [pole]) == 0
    assert ex.spurious_peaks(E, T_true + 0.3 * bump(3.0, 0.05), [pole]) == 1


def test_half_max_width_never_evaluates_outside_the_energy_range():
    """Regression: a broad low-energy resonance made the search probe E < 0 (k = sqrt(E) = NaN)."""
    E0, G = 0.3, 0.2
    seen = []

    def lor(e):
        seen.extend(np.atleast_1d(e).tolist())
        return 1.0 / (1.0 + ((np.asarray(e) - E0) / (G / 2)) ** 2)

    w = ex.half_max_width(lor, E0, 1.0, G, (0.1, 10.0))
    assert min(seen) >= 0.1
    assert w is None or w > 0


def _small(name, **over):
    cfg = json.loads((REPO / "configs" / "phase6" / f"{name}.json").read_text())
    cfg.update(h=0.02, **over)
    return cfg


def test_experiment_runners_complete_end_to_end_on_a_small_config():
    """Smoke test of the full 1D and 3D pipelines (coarse mesh, few points)."""
    d1 = ex.run_rb_1d(_small("rb_1d", n_candidates=300, budgets=[6, 10], random_seeds=[100],
                             scan_uniform=1501, scan_window=101))
    assert {r["design"] for r in d1["rows"]} >= {"greedy", "uniform", "aaa_greedy", "random_s100"}
    assert all("max_fwhm_rel_error" in r["rb_galerkin"] for r in d1["rows"])
    d3 = ex.run_rb_3d(_small("rb_3d", n_candidates=128, n_test=64, n_scans=2, scan_uniform=801,
                             scan_window=101, budgets=[8, 12], random_seeds=[100],
                             n_effectivity=8, effectivity_budgets=[8]))
    assert len(d3["rows"]) == 2 * 3 + 1  # greedy, sobol, random per budget + one tensor grid (n=8)
    assert all(r["rb_galerkin"]["spurious_peaks_total"] >= 0 for r in d3["rows"])
    assert all(s["spurious_peaks_fom"] == 0 for r in d3["rows"] for s in r["rb_galerkin"]["scans"])
