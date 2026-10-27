"""Resonance poles and peaks for Phase 7 (smooth potentials, FEM model).

Definitions used throughout Phase 7:

- **Pole** E_p = E_r - i Gamma_p / 2: a root of 1/t(E) on the unphysical sheet, found by analytic
  continuation of the FEM system to complex E (`AffineFOM.find_pole`). Candidates come from AAA
  rational approximants of the complex amplitude t(k) on overlapping energy windows; each candidate
  is polished by a secant iteration on 1/t and accepted by the scale-free test of
  `AffineFOM.find_pole` (Im E < 0 and |1/t| at the root >= 1e5 times smaller than on the real axis). Gamma_p (the pole width) is *not* the same as the peak FWHM;
  both are reported.
- **Peak**: a local maximum of T(E) with prominence > 0.05 on the evaluation grid, refined by bounded
  maximisation. FWHM = distance between the half-maximum crossings (Brent), defined only when T
  falls below T_peak/2 on both sides within the window; otherwise reported as undefined (broad or
  overlapping resonances on a high background, where FWHM is not a meaningful width).
- **Matching**: a predicted peak matches a reference pole if it lies within 2 Gamma_p of Re E_p.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import AAA

from qscatter.phase7.fom import AffineFOM


def find_poles(
    fom: AffineFOM,
    heights: tuple[float, ...],
    E_range: tuple[float, float],
    window: float = 1.5,
    n_per_window: int = 1500,
    max_width: float = 3.0,
) -> list[complex]:
    lo, hi = E_range
    edges = np.arange(lo, hi + window, window)
    cands: list[complex] = []
    for a, b in zip(edges[:-1], edges[1:]):
        E = np.linspace(max(a - 0.3, 1e-3), min(b, hi) + 0.3, n_per_window)
        mu = np.column_stack([E] + [np.full_like(E, v) for v in heights])
        t, _ = fom.amplitudes(mu)
        try:
            r = AAA(np.sqrt(E), t, rtol=1e-12, max_terms=60)
        except (ValueError, np.linalg.LinAlgError):  # degenerate window (e.g. constant t)
            continue
        for kp, res in zip(r.poles(), r.residues()):
            z = kp**2
            if kp.imag < 0 and abs(res) > 1e-12 and a <= z.real < min(b, hi) and -z.imag * 2 < max_width:
                cands.append(z)
    poles: list[complex] = []
    for z0 in cands:
        z = fom.find_pole(z0, heights)
        if z is None or not (lo <= z.real <= hi):
            continue
        if all(abs(z - p) > 1e-7 * max(1.0, abs(z)) for p in poles):
            poles.append(z)
    return sorted(poles, key=lambda z: z.real)
