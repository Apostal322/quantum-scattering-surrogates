"""Tables and figures for Phase 6 from results/phase6/*/results.json.

    python scripts/phase6/analyze.py

Writes results/phase6/SUMMARY.md and results/phase6/figures/*.png. Recomputes only cheap things:
the example scan of figure 5 (FOM + greedy basis from the stored selection) and a timing of the
exact S-matrix solver for the cost table.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qscatter.phase6 import baselines as bl  # noqa: E402
from qscatter.phase6.fom import FOM, double_barrier_layers  # noqa: E402
from qscatter.phase6.rb import ReducedModel, orthonormalize  # noqa: E402
from qscatter.plotting import INK, INK2, NEUTRAL, SLOTS, plt, setup  # noqa: E402

RES = REPO / "results" / "phase6"
FIG = RES / "figures"


def load(name):
    p = RES / name / "results.json"
    return json.loads(p.read_text()) if p.exists() else None


def e(x):
    return "–" if x is None else f"{x:.1e}"


def rows_by(d, design, n=None):
    return [r for r in d["rows"] if r["design"] == design and (n is None or r["n"] == n)]


def random_designs(d, n):
    return [r for r in d["rows"] if r["design"].startswith("random") and r["n"] == n]


def metric(rows, method, key):
    vals = [r[method][key] for r in rows if method in r]
    return float(np.median(vals)) if vals else None


# ------------------------------------------------------------- resonances


def table_resonances(d, cases):
    """Per-method resonance metrics for (n, design) cases; None-safe."""
    out = ["| N, design | method | resolved | max peak shift / Γ | max FWHM rel. error | FWHM missing* | spurious peaks |",
           "|---|---|---|---|---|---|---|"]
    for n, design, methods in cases:
        rs = rows_by(d, design, n)
        if not rs:
            continue
        for m in methods:
            if m not in rs[0]:
                continue
            x = rs[0][m]
            out.append(f"| {n}, {design} | {m} | {x['resonances_resolved']} | {e(x['max_peak_shift_over_gamma'])} | "
                       f"{e(x.get('max_fwhm_rel_error'))} | {x.get('fwhm_undefined_or_missing', '–')} | {x.get('spurious_peaks_total', '–')} |")
    out.append("\n*Resonances whose FWHM is undefined in the FOM (the peak does not fall to half maximum within "
               "±10Γ, e.g. broad resonances on a high background) or missing in the surrogate. Peak shift and FWHM "
               "error are relative to the FOM (the FOM itself has max |Δt| ≈ 4e-4 against the exact solver).")
    return "\n".join(out)


# ---------------------------------------------------------------------- 1D


def table_1d(d):
    out = ["| N FOM solves | RB-Galerkin greedy | RB-Galerkin uniform | RB-Galerkin random (median of 3) | RB-LSPG uniform | AAA own greedy | AAA uniform | PCHIP(t) uniform | POD regression uniform |",
           "|---|---|---|---|---|---|---|---|---|"]
    for n in sorted({r["n"] for r in d["rows"]}):
        def cell(design, m):
            rs = random_designs(d, n) if design == "random" else rows_by(d, design, n)
            if not rs:
                return "–"
            v = metric(rs, m, "scan_max_abs_dT")
            res = rs[0][m]["resonances_resolved"] if design != "random" else ""
            return f"{e(v)} {res}".strip()
        out.append(f"| {n} | {cell('greedy','rb_galerkin')} | {cell('uniform','rb_galerkin')} | {cell('random','rb_galerkin')} | "
                   f"{cell('uniform','rb_lspg')} | {cell('aaa_greedy','aaa_t')} | {cell('uniform','aaa_t')} | "
                   f"{cell('uniform','pchip_t')} | {cell('uniform','pod_regression')} |")
    return "\n".join(out)


def fig_1d(d):
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ns = sorted({r["n"] for r in d["rows"]})
    series = [
        ("RB-Galerkin, residual greedy", "greedy", "rb_galerkin", SLOTS[0], "-", "o"),
        ("RB-Galerkin, uniform", "uniform", "rb_galerkin", SLOTS[0], "--", "s"),
        ("AAA of t(k), own greedy", "aaa_greedy", "aaa_t", SLOTS[6], "-", "o"),
        ("AAA of t(k), uniform", "uniform", "aaa_t", SLOTS[6], "--", "s"),
        ("PCHIP of t, uniform", "uniform", "pchip_t", SLOTS[1], "--", "s"),
        ("POD coefficient regression, uniform", "uniform", "pod_regression", SLOTS[3], ":", "^"),
    ]
    for label, design, m, c, ls, mk in series:
        xs, ys = [], []
        for n in ns:
            rs = rows_by(d, design, n)
            if rs:
                xs.append(n)
                ys.append(min(max(rs[0][m]["scan_max_abs_dT"], 1e-12), 5.0))  # >1 = blow-up, clipped
        ax.semilogy(xs, ys, ls=ls, marker=mk, color=c, label=label, ms=4)
    ax.axhline(d["fom"]["max_abs_dt"], color=NEUTRAL, lw=0.8, ls=":")
    ax.text(ns[0], d["fom"]["max_abs_dt"] * 0.45, "FOM discretisation error (max |Δt| vs exact S-matrix)",
            ha="left", va="top", fontsize=8, color=INK2)
    ax.set(xlabel="number of high-fidelity (FOM) solves", ylabel="max |T̂ − T_FOM| on resonance-resolved scan",
           title="Stage B, parameter E: projection vs data-driven surrogates", ylim=(1e-10, 10))
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.text(ns[0], 6.0, "values > 5 (interpolant blow-up) drawn at 5", fontsize=7, color=INK2, va="bottom")
    fig.savefig(FIG / "p6_1d_error_vs_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- 3D


def table_3d(d):
    out = ["| N | RB-Galerkin greedy: test max\\|Δt\\| / scan max\\|ΔT\\| / resonances | RB-Galerkin Sobol | RB-Galerkin random (median) | RB-Galerkin tensor | RB-LSPG greedy | RBF on t, best design† | POD regression, best design† |",
           "|---|---|---|---|---|---|---|---|"]
    for n in sorted({r["n"] for r in d["rows"]}):
        def trip(rs, m):
            if not rs:
                return "–"
            a, b = metric(rs, m, "test_max_abs_dt"), metric(rs, m, "scan_max_abs_dT")
            res = rs[0][m]["resonances_resolved"] if len(rs) == 1 else ""
            return f"{e(a)} / {e(b)} {res}".strip()
        def best(m):
            cands = [r for r in d["rows"] if r["n"] == n]
            r = min(cands, key=lambda r: r[m]["scan_max_abs_dT"])
            return f"{e(r[m]['test_max_abs_dt'])} / {e(r[m]['scan_max_abs_dT'])} ({r['design']})"
        out.append(f"| {n} | {trip(rows_by(d,'greedy',n),'rb_galerkin')} | {trip(rows_by(d,'sobol',n),'rb_galerkin')} | "
                   f"{trip(random_designs(d,n),'rb_galerkin')} | {trip(rows_by(d,'tensor',n),'rb_galerkin')} | "
                   f"{trip(rows_by(d,'greedy',n),'rb_lspg')} | {best('rbf_t')} | {best('pod_regression')} |")
    return "\n".join(out)


def fig_3d(d):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    ns = sorted({r["n"] for r in d["rows"]})
    for ax, key, title in [(axs[0], "test_max_abs_dt", "2048 held-out random (E, V_L, V_R)"),
                           (axs[1], "scan_max_abs_dT", "6 resonance-resolved E scans (max |ΔT|)")]:
        for design, c, ls, label in [("greedy", SLOTS[0], "-", "residual greedy"), ("sobol", SLOTS[2], "--", "Sobol"),
                                     ("random", NEUTRAL, ":", "random (median of 3)"), ("tensor", SLOTS[3], "", "tensor grid")]:
            xs, ys = [], []
            for n in ns:
                rs = random_designs(d, n) if design == "random" else rows_by(d, design, n)
                if rs:
                    xs.append(n)
                    ys.append(max(metric(rs, "rb_galerkin", key), 1e-12))
            ax.semilogy(xs, ys, ls=ls or "none", marker="o", ms=4, color=c, label=f"RB-Galerkin, {label}")
        for m, c, label in [("rbf_t", SLOTS[1], "RBF on t (best design†)"), ("pod_regression", SLOTS[4], "POD regression (best design†)")]:
            ys = [min(r[m][key] for r in d["rows"] if r["n"] == n) for n in ns]
            ax.semilogy(ns, ys, ls="-.", marker="^", ms=4, color=c, label=label)
        ax.set(xlabel="number of FOM solves", title=title)
    axs[0].set_ylabel("max |t̂ − t_FOM| (left) / max |T̂ − T_FOM| (right)")
    fig.text(0.5, -0.04, "† best of the 6 sampling designs, chosen on the test data: optimistic for the baseline",
             ha="center", fontsize=8, color=INK2)
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Double barrier with parameters (E, V_L, V_R): error vs high-fidelity budget", fontsize=11, y=1.03)
    fig.savefig(FIG / "p6_3d_error_vs_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_effectivity(d):
    eff = d["effectivity"]
    fig, axs = plt.subplots(1, len(eff), figsize=(3.0 * len(eff), 3.1), sharey=True)
    inv = np.array(d["inverse_norm"])
    for ax, row in zip(np.atleast_1d(axs), eff):
        res, et, ep = map(np.array, (row["residual"], row["err_t"], row["err_psi"]))
        sc = ax.scatter(res, et, c=np.log10(inv), s=6, cmap="viridis")
        ax.set(xscale="log", yscale="log", xlabel="residual ||b − A V c||", title=f"n = {row['n']}: ρ = {row['spearman_residual_vs_err_t']:.2f}")
    np.atleast_1d(axs)[0].set_ylabel("true error |t̂ − t|")
    fig.colorbar(sc, ax=axs, label="log10 ||A(μ)⁻¹||")
    fig.suptitle("Residual indicator vs true error on 256 held-out points (greedy bases)", fontsize=10, y=1.06)
    fig.savefig(FIG / "p6_residual_vs_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def table_effectivity(d):
    out = ["| n | Spearman ρ(residual, \\|Δt\\|) | ρ(residual, \\|\\|Δψ\\|\\|) | bound ‖A⁻¹‖·‖res‖ ≥ ‖Δψ‖ holds | bound / true error: median, max |",
           "|---|---|---|---|---|"]
    for r in d["effectivity"]:
        out.append(f"| {r['n']} | {r['spearman_residual_vs_err_t']:.2f} | {r['spearman_residual_vs_err_psi']:.2f} | "
                   f"{100*r['bound_holds_fraction']:.0f} % | {r['bound_effectivity_median']:.1f}, {r['bound_effectivity_max']:.1e} |")
    inv = np.array(d["inverse_norm"])
    out.append(f"\n‖A(μ)⁻¹‖ on these points: median {np.median(inv):.0f}, max {inv.max():.1e}.")
    return "\n".join(out)


def fig_example_scan(d):
    """Hardest scan: RB with 16 greedy snapshots vs RBF and POD regression on the same 16 points."""
    cfg = d["provenance"]["config"]
    fom = FOM(double_barrier_layers(cfg["barrier_width"], cfg["well_width"]), cfg["h"])
    lo, hi = np.array(cfg["lo"]), np.array(cfg["hi"])
    sel = np.array(d["greedy"]["selected"])
    n = 16
    mu_tr = sel[:n]
    S = fom.solve(mu_tr).T
    t_tr, _ = fom.observables(mu_tr, S[0], S[-1])
    rm = ReducedModel(fom, orthonormalize(S)[0])
    # the narrowest resonance over all held-out scans (smallest |Im E_pole|)
    sc, (zr, zi) = max(((s, z) for s in d["scans"] for z in s["poles"]), key=lambda sz: sz[1][1])
    G = -2 * zi
    E = np.linspace(zr - 6 * G, zr + 6 * G, 1201)
    mu = np.column_stack([E, np.full_like(E, sc["heights"]["VL"]), np.full_like(E, sc["heights"]["VR"])])
    T = np.abs(fom.amplitudes(mu)[0]) ** 2
    T_rb = np.abs(rm.amplitudes(mu)[0]) ** 2
    T_rbf = np.abs(bl.direct_rbf(mu_tr, t_tr, mu, lo, hi)) ** 2
    T_pod = np.abs(bl.pod_regression(fom, mu_tr, S, mu, lo, hi)) ** 2
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(E, T, color=INK, lw=2.4, label="FOM (truth)")
    ax.plot(E, T_rb, color=SLOTS[0], lw=1.2, ls="--", label=f"RB-Galerkin, {n} greedy snapshots")
    ax.plot(E, np.clip(T_rbf, 0, 1.2), color=SLOTS[1], lw=1.0, label="RBF on t, same 16 points")
    ax.plot(E, np.clip(T_pod, 0, 1.2), color=SLOTS[4], lw=1.0, ls=":", label="POD regression, same 16 points")
    h = sc["heights"]
    ax.set(xlabel="E", ylabel="T(E)", title=f"Narrowest resonance of a held-out configuration (V_L={h['VL']:.2f}, V_R={h['VR']:.2f}, Γ={G:.1e})")
    ax.legend(fontsize=7.5, frameon=False)
    fig.savefig(FIG / "p6_example_resonance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"heights": h, "gamma": G, "max_dT_rb": float(np.max(np.abs(T_rb - T))),
            "max_dT_rbf": float(np.max(np.abs(T_rbf - T))), "max_dT_pod": float(np.max(np.abs(T_pod - T)))}


# ------------------------------------------------------------------ n-width


def table_nwidth(d):
    out = ["| case | resonances in [0.1, 10] | narrowest Γ | rank (σ/σ₁ > 1e-6) | rank (> 1e-10) |", "|---|---|---|---|---|"]
    for c in d["cases"]:
        g = min(c["gammas"]) if c["gammas"] else None
        out.append(f"| {c['label']} | {c['n_poles_in_range']} | {e(g)} | {c['rank_1e-6']} | {c['rank_1e-10']} |")
    return "\n".join(out)


def fig_nwidth(d):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.7), sharey=True)
    vcases = [c for c in d["cases"] if c["label"].startswith("V0=")]
    dcases = [c for c in d["cases"] if c["label"].startswith("d=") or c["label"] == "V0=10"]
    for ax, cases, title in [(axs[0], vcases, "barrier height (resonance widths change ×50)"),
                             (axs[1], sorted(dcases, key=lambda c: c["well_width"]), "well width (number of resonances changes)")]:
        for i, c in enumerate(cases):
            s = np.array(c["singular_values"])
            lab = c["label"] if not c["label"].startswith("V0=10") or ax is axs[0] else "d=2.5"
            ax.semilogy(np.arange(1, s.size + 1), s, marker=".", ms=3, color=SLOTS[i % len(SLOTS)],
                        label=f"{lab}: {c['n_poles_in_range']} res., Γmin={min(c['gammas']):.0e}")
        ax.set(xlabel="index i", title=title, xlim=(0, 30), ylim=(1e-16, 2))
        ax.legend(fontsize=7, frameon=False)
    axs[0].set_ylabel("σᵢ / σ₁ of the snapshot matrix")
    fig.savefig(FIG / "p6_singular_values.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- costs


def table_cost():
    p = RES / "cost" / "cost.md"
    return p.read_text().split("\n", 2)[2] if p.exists() else "Run scripts/phase6/benchmark_cost.py first."


def main():
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    d1, d3, dn = load("rb_1d"), load("rb_3d"), load("nwidth")
    parts = ["# Phase 6 results (generated by scripts/phase6/analyze.py)\n"]
    if d1:
        fig_1d(d1)
        parts += ["## 1D: Stage B, parameter E\n",
                  f"FOM: {d1['fom']['nodes']} nodes; FOM vs exact S-matrix on the scan: max |Δt| = {e(d1['fom']['max_abs_dt'])}, "
                  f"RMSE(T) = {e(d1['fom']['rmse_T'])}. Test: {d1['test_points']} resonance-resolved energies "
                  "(never used for training or selection). Cells: max |T̂ − T_FOM| and resonances resolved.\n",
                  table_1d(d1), "", "### Resonance metrics (1D)\n",
                  table_resonances(d1, [(n, dz, ["rb_galerkin", "aaa_t", "pchip_t", "pod_regression"])
                                        for n in (10, 15, 20) for dz in ("greedy", "aaa_greedy", "uniform")]), ""]
    if d3:
        fig_3d(d3)
        fig_effectivity(d3)
        ex = fig_example_scan(d3)
        parts += ["## 3D: parameters (E, V_L, V_R)\n", "Cells: max |Δt| on 2048 held-out random points / max |ΔT| on 6 "
                  "resonance-resolved scans (held-out heights), and resonances resolved on the scans.\n",
                  table_3d(d3), "\n† best of the 6 sampling designs, chosen on the test data (optimistic for the baseline).\n",
                  "### Resonance metrics (3D, 6 held-out scans)\n",
                  table_resonances(d3, [(n, "greedy", ["rb_galerkin", "rb_lspg", "rbf_t", "pod_regression"]) for n in (12, 16, 24)]),
                  "", "### Residual indicator vs true error\n", table_effectivity(d3), "",
                  f"Example (figure p6_example_resonance): narrowest resonance Γ = {ex['gamma']:.1e}; max |ΔT| with the same 16 "
                  f"FOM solves: RB {e(ex['max_dT_rb'])}, RBF {e(ex['max_dT_rbf'])}, POD regression {e(ex['max_dT_pod'])}.\n"]
        parts += ["## Cost (scripts/phase6/benchmark_cost.py)\n", table_cost(), ""]
    if dn:
        fig_nwidth(dn)
        parts += ["## What sets the basis size (snapshot singular values)\n", table_nwidth(dn), ""]
    (RES / "SUMMARY.md").write_text("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
