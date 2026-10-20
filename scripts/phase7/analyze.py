"""Tables and figures for Phase 7 from results/phase7/*/results.json (and cost/cost.json).

    python scripts/phase7/analyze.py

Writes results/phase7/SUMMARY.md and results/phase7/figures/*.png. Pure post-processing: nothing
is recomputed, and missing experiments are skipped (and listed as missing in the summary).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qscatter.plotting import INK2, NEUTRAL, SLOTS, plt, setup  # noqa: E402

RES = REPO / "results" / "phase7"
FIG = RES / "figures"


def load(name, file="results.json"):
    p = RES / name / file
    return json.loads(p.read_text()) if p.exists() else None


def e(x, f="{:.1e}"):
    return "–" if x is None else f.format(x)


def med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else None


# --------------------------------------------------------------------------- A


def section_validation(d, out):
    out += ["## A. Numerical validation (FEM vs independent references)", ""]
    for c in d["configs"]:
        hh = c["heights"]
        out += [f"### Heights V_L = {hh[0]}, V_R = {hh[1]}", "",
                "ODE self-consistency (max |Δt| vs rtol 1e-12): " + ", ".join(f"rtol {k}: {e(v)}" for k, v in c["ode_tolerance"].items()),
                f"ODE flux defect: {e(c['ode_flux_defect'])}", "",
                "Domain truncation (ODE, max |Δt| vs the largest domain): " + ", ".join(f"{k}: {e(v)}" for k, v in c["truncation"].items()), "",
                "| h | nodes | max abs Δt | max abs Δr | max abs ΔT | MAE T | flux defect | s / solve | poles | max abs ΔE_r / Γ | max rel. err Γ |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for m in c["mesh"]:
            pe = [p for p in m["pole_errors"] if p]
            out.append(f"| {m['h']} | {m['nodes']} | {e(m['max_abs_dt'])} | {e(m['max_abs_dr'])} | {e(m['max_abs_dT'])} | "
                       f"{e(m['mae_T'])} | {e(m['flux_defect'])} | {e(m['seconds_per_solve'])} | {len(m['poles'])} | "
                       f"{e(max((p['err_Er_over_gamma'] for p in pe), default=None))} | {e(max((p['rel_err_gamma'] for p in pe), default=None))} |")
        out += ["", "Reference poles (Richardson, two finest meshes): " +
                ", ".join(f"{p[0]:.6f} − {-p[1]:.3e} i" for p in c["reference_poles"]),
                "", "Pole shift when the domain grows (rel_cut × 1e-2): " + ", ".join(e(x) for x in c["pole_shift_domain_change"]), "",
                "| energy grid points | spacing | peaks found | poles with a peak |", "|---|---|---|---|"]
        for gr in c["energy_grid"]:
            out.append(f"| {gr['n']} | {gr['spacing']:.2e} | {gr['peaks']} | {gr['poles_with_a_peak']} / {len(c['reference_poles'])} |")
        out.append("")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, c in enumerate(d["configs"]):
        hs = [m["h"] for m in c["mesh"]]
        ax[0].loglog(hs, [m["max_abs_dt"] for m in c["mesh"]], "o-", color=SLOTS[i], label=f"V = {c['heights']}")
        pe = [max((p["err_Er_over_gamma"] for p in m["pole_errors"] if p), default=np.nan) for m in c["mesh"]]
        ax[1].loglog(hs, pe, "o-", color=SLOTS[i], label=f"V = {c['heights']}")
    h = np.array(hs)
    for a in ax:
        a.loglog(h, (h / h[0]) ** 2 * a.get_lines()[0].get_ydata()[0], "--", color=NEUTRAL, label="∝ h²")
        a.set_xlabel("mesh size h")
        a.legend()
    ax[0].set_ylabel("max |t_FEM − t_ODE|")
    ax[0].set_title("FEM vs ODE on the real axis")
    ax[1].set_ylabel("max |ΔE_r| / Γ (vs Richardson)")
    ax[1].set_title("Pole positions")
    fig.savefig(FIG / "A_mesh_convergence.png")
    plt.close(fig)


# --------------------------------------------------------------------- B / C / E


def section_generalization(d, out):
    rows = d["rows"]
    budgets = sorted({r["n"] for r in rows})
    out += ["## B/C. Generalisation and fair comparison (3D: E, V_L, V_R)", "",
            f"FOM: {d['fom']['nodes']} nodes, h = {d['fom']['h']}, X = {d['fom']['X']}. RBF kernel chosen on validation: "
            f"**{d['rbf_kernel']}** (validation RMSE of t: " + ", ".join(f"{k} {e(v)}" for k, v in d["rbf_kernel_validation"].items()) + ").",
            f"Data-driven baselines on the pre-declared design `{d['manifest']['baseline_design']}` only.", ""]
    for split, title in (("test", "in-domain test (1024 points)"), ("test_ood", "out-of-domain test (512 points, extrapolation)")):
        out += [f"### Max |Δt| on the {title}", "",
                "| n | RB greedy (Galerkin) | RB greedy (LSPG) | RB Sobol | RB random (median of 3) | RBF (Sobol) | POD regression (Sobol) |",
                "|---|---|---|---|---|---|---|"]
        for n in budgets:
            g = [r for r in rows if r["n"] == n and r["design"] == "greedy"][0]
            sb = [r for r in rows if r["n"] == n and r["design"] == "sobol"][0]
            rnd = [r for r in rows if r["n"] == n and r["design"].startswith("random")]
            out.append(f"| {n} | {e(g['rb_galerkin'][split]['max_abs_dt'])} | {e(g['rb_lspg'][split]['max_abs_dt'])} | "
                       f"{e(sb['rb_galerkin'][split]['max_abs_dt'])} | {e(med([r['rb_galerkin'][split]['max_abs_dt'] for r in rnd]))} | "
                       f"{e(sb['rbf_t'][split]['max_abs_dt'])} | {e(sb['pod_regression'][split]['max_abs_dt'])} |")
        out += ["", "| MLP: n (own Sobol design) | chosen (width, lr) | max abs Δt | MAE T | fit s |", "|---|---|---|---|---|"]
        for m in d["mlp"]:
            out.append(f"| {m['n']} | {m['choice']['width']}, {m['choice']['lr']} | {e(m[split]['max_abs_dt'])} | {e(m[split]['mae_T'])} | {m['fit_seconds']:.0f} |")
        out.append("")
    out += ["### Resonance scans (test: 6 height pairs; OOD: 4)", "",
            "| n | method | split | recall | max peak shift / Γ | max FWHM rel. err | spurious peaks (FOM) |", "|---|---|---|---|---|---|---|"]
    for n in budgets:
        g = [r for r in rows if r["n"] == n and r["design"] == "greedy"][0]
        sb = [r for r in rows if r["n"] == n and r["design"] == "sobol"][0]
        for name, res in (("RB greedy", g["rb_galerkin"]), ("RB Sobol", sb["rb_galerkin"]), ("RBF Sobol", sb["rbf_t"]), ("POD reg. Sobol", sb["pod_regression"])):
            for split in ("scans_test", "scans_ood"):
                s = res[split]
                out.append(f"| {n} | {name} | {split[6:]} | {e(s['recall'], '{:.2f}')} | {e(s['max_peak_shift_over_gamma'])} | "
                           f"{e(s['max_fwhm_rel_error'])} | {s['spurious_peaks']} ({s['spurious_peaks_fom']}) |")
    for m in d["mlp"]:
        for split in ("scans_test", "scans_ood"):
            s = m[split]
            out.append(f"| {m['n']} | MLP | {split[6:]} | {e(s['recall'], '{:.2f}')} | {e(s['max_peak_shift_over_gamma'])} | "
                       f"{e(s['max_fwhm_rel_error'])} | {s['spurious_peaks']} ({s['spurious_peaks_fom']}) |")
    out += ["", "### B3. Rectangular-barrier basis transferred to the Gaussian problem", "",
            "| rectangular snapshots | Gaussian snapshots | r | max abs Δt test | max abs Δt OOD | scan recall (test) |", "|---|---|---|---|---|---|"]
    for t in d["transfer"]:
        r = t["result"]
        out.append(f"| {t['n_rect']} | {t['n_gauss']} | {t['r']} | {e(r['test']['max_abs_dt'])} | {e(r['test_ood']['max_abs_dt'])} | {e(r['scans_test']['recall'], '{:.2f}')} |")
    pw = d["power_iteration_vs_exact_inverse_norm"]
    out += ["", "## E. Residual reliability (greedy RB)", "",
            "Certified bound: ‖e_ψ‖ ≤ ‖res‖ / σ_min(A), σ_min from the banded Golub–Kahan eigenproblem minus a "
            "floating-point margin. Power iteration (Phase 6) gives a *lower* estimate of ‖A⁻¹‖: "
            f"its product with the certified σ_min ranges over [{pw['min_ratio_lower_over_exact']:.6f}, {pw['max_ratio_lower_over_exact']:.6f}], "
            "so it cannot serve as a bound.", "",
            "| n | points | Spearman(res, ‖e_ψ‖) | Spearman(res, abs e_t) | bound holds (ψ) | bound holds (t)* | median bound/‖e_ψ‖ | median bound/abs e_t |",
            "|---|---|---|---|---|---|---|---|"]
    for ef in d["effectivity"]:
        for lab in ("random", "near_resonance"):
            b = ef[lab]
            out.append(f"| {ef['n']} | {lab} ({b['n_points']}) | {b['spearman_res_vs_err_psi']:.2f} | {b['spearman_res_vs_err_t']:.2f} | "
                       f"{b['certified_bound_holds_psi']:.3f} | {b['certified_bound_holds_t']:.3f} | {e(b['bound_over_err_psi_median'])} | {e(b['bound_over_err_t_median'])} |")
    out += ["", "*The bound is proved for the wave-function error. Since |e_t| = |e_ψ(L)| ≤ ‖e_ψ‖_∞ ≤ ‖e_ψ‖_2, it also bounds the error of t.", ""]

    # figures
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, split, title in ((ax[0], "test", "in-domain test"), (ax[1], "test_ood", "out-of-domain (extrapolation)")):
        def series(design, method):
            return [med([r[method][split]["max_abs_dt"] for r in rows if r["n"] == n and r["design"].startswith(design) and method in r]) for n in budgets]
        a.semilogy(budgets, series("greedy", "rb_galerkin"), "o-", color=SLOTS[0], label="RB greedy")
        a.semilogy(budgets, series("sobol", "rb_galerkin"), "s-", color=SLOTS[2], label="RB Sobol")
        a.semilogy(budgets, series("random", "rb_galerkin"), "^-", color=NEUTRAL, label="RB random (median)")
        a.semilogy(budgets, series("sobol", "rbf_t"), "d-", color=SLOTS[1], label="RBF on t (Sobol)")
        a.semilogy(budgets, series("sobol", "pod_regression"), "v-", color=SLOTS[3], label="POD regression (Sobol)")
        a.semilogy([m["n"] for m in d["mlp"]], [m[split]["max_abs_dt"] for m in d["mlp"]], "x--", color=SLOTS[4], label="MLP (own Sobol design)")
        a.set_xscale("log")
        a.set_xlabel("FOM queries n")
        a.set_title(title)
    ax[0].set_ylabel("max |t̂ − t_FEM|")
    ax[0].legend(fontsize=8)
    fig.savefig(FIG / "BC_error_vs_queries.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, ef in enumerate(d["effectivity"]):
        for a, lab in zip(ax, ("random", "near_resonance")):
            b = ef[lab]
            res, err, sm = np.array(b["residual"]), np.array(b["err_psi"]), np.array(b["sigma_min"])
            a.loglog(err, res / sm, ".", color=SLOTS[i], alpha=0.6, label=f"n = {ef['n']}")
    for a, lab in zip(ax, ("random test points", "within ±2Γ of resonances")):
        lim = [min(a.get_xlim()[0], a.get_ylim()[0]), max(a.get_xlim()[1], a.get_ylim()[1])]
        a.loglog(lim, lim, "--", color=INK2, label="bound = error")
        a.set_xlabel("true error ‖ψ̂ − ψ‖")
        a.set_ylabel("certified bound ‖res‖ / σ_min")
        a.set_title(lab)
        a.legend(fontsize=8)
    fig.savefig(FIG / "E_certified_bound.png")
    plt.close(fig)


def section_slice(d, out):
    methods = ["pchip_uniform", "aaa_uniform", "aaa_adaptive", "rb_greedy"]
    budgets = [r["n"] for r in d["scans"][0]["rows"]]
    out += ["## C-1D. Energy-only methods vs RB on the test scans (heights fixed)", "",
            "Median over the 6 test scans; equal numbers n of FOM queries per scan.", "",
            "| n | method | median max abs Δt | worst max abs Δt | mean recall | median max ΔE_r/Γ (poles) | poles matched | unmatched AAA poles |",
            "|---|---|---|---|---|---|---|---|"]
    for i, n in enumerate(budgets):
        for m in methods:
            rs = [sc["rows"][i][m] for sc in d["scans"]]
            pe = [p for r in rs if r["pole_errors"] for p in r["pole_errors"]]
            matched = [p["matched"] for p in pe]
            errs = [max((p.get("err_Er_over_gamma", np.inf) for p in r["pole_errors"]), default=None) for r in rs if r["pole_errors"]]
            rec = [np.mean([x["resolved"] for x in r["resonances"]]) for r in rs if r["resonances"]]
            un = [r.get("unmatched_poles_near_axis") for r in rs]
            out.append(f"| {n} | {m} | {e(med([r['max_abs_dt'] for r in rs]))} | {e(max(r['max_abs_dt'] for r in rs))} | "
                       f"{e(float(np.mean(rec)) if rec else None, '{:.2f}')} | {e(med(errs))} | "
                       f"{(f'{sum(matched)}/{len(matched)}') if matched else '–'} | {sum(u for u in un if u is not None) if any(u is not None for u in un) else '–'} |")
    out.append("")
    fig, ax = plt.subplots(figsize=(6, 4))
    for c, m in zip(SLOTS, methods):
        ax.semilogy(budgets, [med([sc["rows"][i][m]["max_abs_dt"] for sc in d["scans"]]) for i in range(len(budgets))], "o-", color=c, label=m)
    ax.set_xlabel("FOM queries per scan")
    ax.set_ylabel("median over scans of max |Δt|")
    ax.set_title("One energy line, heights fixed")
    ax.legend()
    fig.savefig(FIG / "C1D_energy_line.png")
    plt.close(fig)


# --------------------------------------------------------------------------- D


def section_basis(d, out):
    out += ["## D. Basis size vs resonance structure (fixed-height energy lines)", "",
            "| sweep | value | resonances | Γ_min | snapshot rank 1e-6 / 1e-8 | n for 1e-2 | 1e-4 | 1e-6 | 1e-8 |", "|---|---|---|---|---|---|---|---|---|"]
    for knob, rs in d["sweeps"].items():
        for r in rs:
            nf = r["n_for_accuracy"]
            out.append(f"| {knob} | {r[knob]} | {r['n_resonances']} | {e(r['gamma_min'])} | {r['snapshot_rank']['1e-06']} / {r['snapshot_rank']['1e-08']} | "
                       f"{nf['0.01']} | {nf['0.0001']} | {nf['1e-06']} | {nf['1e-08']} |")
    out += ["", "Descriptive least-squares fit n ≈ a + b·N_res + c·log10(1/Γ_min) over all configurations:", "",
            "| tolerance | a, b, c | R² | corr(n, N_res) | corr(n, log 1/Γ) | corr(N_res, log 1/Γ) | configs |", "|---|---|---|---|---|---|---|"]
    for tol, f in d["fits"].items():
        out.append(f"| {tol} | " + ", ".join(f"{x:.2f}" for x in f["coef_const_Nres_log10invGamma"]) +
                   f" | {f['r2']:.2f} | {f['corr_n_vs_Nres']:.2f} | {f['corr_n_vs_log_invGamma']:.2f} | {f['corr_Nres_vs_log_invGamma']:.2f} | {f['n_configs']} |")
    out.append("")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for c, (knob, rs) in zip(SLOTS, d["sweeps"].items()):
        n6 = [r["n_for_accuracy"]["1e-06"] for r in rs]
        ax[0].plot([r["n_resonances"] for r in rs], n6, "o-", color=c, label=f"sweep {knob}")
        ax[1].plot([np.log10(1 / r["gamma_min"]) if r["gamma_min"] else np.nan for r in rs], n6, "o-", color=c, label=f"sweep {knob}")
    ax[0].set_xlabel("resonances in the window")
    ax[1].set_xlabel("log10(1 / Γ_min)")
    for a in ax:
        a.set_ylabel("greedy n for max |Δt| < 1e-6")
        a.legend(fontsize=8)
    fig.savefig(FIG / "D_basis_size.png")
    plt.close(fig)


# --------------------------------------------------------------------------- F


def section_robustness(d, out):
    out += ["## F. Robustness (n = 32 queries per energy line unless stated)", ""]
    for L in d["lines"]:
        out += [f"### Heights {L['heights']}", "", "Noise (median over seeds of max |Δt|; recall; unmatched AAA poles summed over seeds):", "",
                "| σ | AAA uniform | AAA adaptive | PCHIP | RB (noisy snapshots) |", "|---|---|---|---|---|"]
        for sig in sorted({r["sigma"] for r in L["noise"]}):
            rs = [r for r in L["noise"] if r["sigma"] == sig]

            def cell(k):
                v = [r[k] for r in rs]
                un = [x.get("unmatched_poles_near_axis") for x in v]
                s = f"{e(med([x['max_abs_dt'] for x in v]))}; {e(med([x['recall'] for x in v]), '{:.2f}')}"
                return s + (f"; {sum(un)}" if un[0] is not None else "")
            out.append(f"| {sig:.0e} | {cell('aaa_uniform')} | {cell('aaa_adaptive')} | {cell('pchip_uniform')} | {cell('rb_greedy_noisy_snapshots')} |")
        out += ["", "| coarse h | FEM(h) vs FEM | AAA on coarse t | RB from coarse snapshots, fine operator |", "|---|---|---|---|"]
        for r in L["solver_inaccuracy"]:
            out.append(f"| {r['h_coarse']} | {e(r['fom_coarse_vs_fine']['max_abs_dt'])} | {e(r['aaa_uniform_coarse_data']['max_abs_dt'])} | {e(r['rb_coarse_snapshots_fine_operator']['max_abs_dt'])} |")
        out += ["", "| jitter δ | AAA | PCHIP | RB on the same design |", "|---|---|---|---|"]
        for dl in sorted({r["delta"] for r in L["jitter"]}):
            rs = [r for r in L["jitter"] if r["delta"] == dl]
            out.append(f"| {dl} | " + " | ".join(f"{e(med([r[k]['max_abs_dt'] for r in rs]))} (worst {e(max(r[k]['max_abs_dt'] for r in rs))})" for k in ("aaa", "pchip", "rb_same_design")) + " |")
        p = L["precision"]
        out += ["", f"Online solve precision: complex128 {e(p['complex128'])}, complex64 {e(p['complex64'])}; max cond of the reduced matrix {e(p['cond_reduced_max'])}.", ""]
    out += ["### Near-degenerate resonances (symmetric triple Gaussian, two coupled wells)", "",
            "| middle height | closest pair: splitting / mean Γ | n | RB greedy max abs Δt; recall | AAA adaptive | AAA uniform | AAA poles matched (adaptive) |", "|---|---|---|---|---|---|---|"]
    for c in d["doublets"]["configs"]:
        sp = min(c["adjacent_pairs"], key=lambda q: q["splitting_over_mean_gamma"]) if c["adjacent_pairs"] else None
        for r in c["rows"]:
            out.append(f"| {c['heights'][1]} | {e(sp['splitting_over_mean_gamma'], '{:.2f}') if sp else '–'} | {r['n']} | "
                       f"{e(r['rb_greedy']['max_abs_dt'])}; {e(r['rb_greedy']['recall'], '{:.2f}')} | "
                       f"{e(r['aaa_adaptive']['max_abs_dt'])}; {e(r['aaa_adaptive']['recall'], '{:.2f}')} | "
                       f"{e(r['aaa_uniform']['max_abs_dt'])}; {e(r['aaa_uniform']['recall'], '{:.2f}')} | {e(r['aaa_adaptive'].get('poles_matched'), '{:.2f}')} |")
    out.append("")
    L = d["lines"][0]
    fig, ax = plt.subplots(figsize=(6, 4))
    sigs = sorted({r["sigma"] for r in L["noise"]})
    for c, k in zip(SLOTS, ("aaa_uniform", "aaa_adaptive", "pchip_uniform", "rb_greedy_noisy_snapshots")):
        ax.loglog(sigs, [med([r[k]["max_abs_dt"] for r in L["noise"] if r["sigma"] == s]) for s in sigs], "o-", color=c, label=k)
    ax.loglog(sigs, sigs, "--", color=NEUTRAL, label="error = σ")
    ax.set_xlabel("relative noise σ")
    ax.set_ylabel("median max |Δt| (5 seeds)")
    ax.set_title(f"Noise, heights {L['heights']}")
    ax.legend(fontsize=8)
    fig.savefig(FIG / "F_noise.png")
    plt.close(fig)


def section_cost(c, out):
    f = c["fom"]
    out += ["## Cost and break-even", "",
            f"Hardware: {c['hardware']['cpu']}, 1 thread (of {c['hardware']['logical_cores']} logical cores); "
            f"numpy {c['hardware']['numpy']}, scipy {c['hardware']['scipy']}.", "",
            f"FEM: N = {f['nodes']}, single query {e(f['single_s'])} s, batched {e(f['batched_s'])} s/point, error vs ODE {e(f['max_abs_dt_vs_ode'])}.",
            f"ODE (DOP853, rtol 1e-10): {e(c['ode_single_s'])} s per energy.",
            f"Staircase S-matrix with FEM-equivalent accuracy: M = {c['staircase']['M_matched']} steps, {e(c['staircase']['single_s'])} s per energy "
            "(levels: " + ", ".join(f"M={x['M']}: {e(x['max_abs_dt_vs_ode'])}" for x in c["staircase"]["levels"]) + "); "
            f"batched over 1024 energies at fixed heights: {e(c['staircase'].get('batched_1024_energies_fixed_heights_per_energy_s'))} s per energy.", "",
            "| n | greedy offline s | single query s | batched s/pt | batched + residual s/pt | break-even vs FEM (greedy / Sobol design) | vs ODE | vs staircase |",
            "|---|---|---|---|---|---|---|---|"]
    for r in c["rb"]:
        b = r["break_even_queries"]
        out.append(f"| {r['n']} | {r['greedy_total_s']:.2f} | {e(r['rb_single_s'])} | {e(r['rb_batched_s'])} | {e(r['rb_batched_with_residual_s'])} | "
                   f"{b['vs_fem']:.0f} / {b['vs_fem_sobol_design']:.0f} | {b['vs_ode']:.0f} | {e(b.get('vs_staircase_smatrix'), '{:.0f}')} |")
    out += ["", f"RBF (32 points): fit {e(c['rbf_32']['fit_s'])} s, {e(c['rbf_32']['eval_per_point_s'])} s/point. "
            f"AAA (32 points): fit {e(c['aaa_32']['fit_s'])} s, {e(c['aaa_32']['eval_per_point_s'])} s/point.", ""]


def main():
    setup()
    FIG.mkdir(parents=True, exist_ok=True)
    out = ["# Phase 7 results summary (generated by scripts/phase7/analyze.py)", ""]
    missing = []
    for name, fn in (("validation", section_validation), ("generalization", section_generalization),
                     ("slice1d", section_slice), ("basis", section_basis), ("robustness", section_robustness)):
        d = load(name)
        if d is None:
            missing.append(name)
            continue
        out.append(f"<!-- {name}: commit {d['provenance']['git_commit'][:7]}, modified since commit: "
                   f"{d['provenance']['code_modified_since_commit']}, wall {d['wall_seconds']:.0f} s -->")
        fn(d, out)
    c = load("cost", "cost.json")
    if c is None:
        missing.append("cost")
    else:
        section_cost(c, out)
    if missing:
        out += ["", "Missing experiments: " + ", ".join(missing)]
    (RES / "SUMMARY.md").write_text("\n".join(out) + "\n")
    print(f"wrote {RES / 'SUMMARY.md'}; missing: {missing}")


if __name__ == "__main__":
    main()
