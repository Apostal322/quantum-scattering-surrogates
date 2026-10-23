"""Frozen evaluation protocol for Phase 7 (no-leakage rules, split manifests).

Rules (checked by tests/phase7/test_protocol.py):
1. Every set (greedy candidates, sampling designs, validation, in-domain test, out-of-domain test,
   resonance scans) is generated from its own declared seed; the manifest stores seeds, domains,
   sizes and a SHA-256 of the generated parameters.
2. Hyperparameters (RBF kernel, MLP architecture/learning rate/epochs) are chosen on the
   validation set only. The test sets are evaluated once, after all choices are frozen.
3. Data-driven baselines use a design declared in the config *before* any evaluation
   (`baseline_design`). No "best design" is ever selected on test data. (Phase 6 did select the
   displayed RBF/POD design on test data; that historical limitation is documented, not repeated.)
4. Out-of-domain test points lie outside the training box in at least one height and are reported
   separately as extrapolation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc


def _hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def sobol(n: int, lo, hi, seed: int) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return qmc.scale(qmc.Sobol(len(lo), seed=seed).random(n), lo, hi)


@dataclass
class Splits:
    candidates: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    test_ood: np.ndarray
    ood_labels: list[str]
    scan_heights_val: list[list[float]]
    scan_heights_test: list[list[float]]
    scan_heights_ood: list[list[float]]
    manifest: dict[str, Any]


def make_splits(cfg: dict) -> Splits:
    lo, hi = np.array(cfg["train_lo"], float), np.array(cfg["train_hi"], float)
    sd = cfg["seeds"]
    cand = sobol(cfg["n_candidates"], lo, hi, sd["candidates"])
    val = sobol(cfg["n_validation"], lo, hi, sd["validation"])
    test = sobol(cfg["n_test"], lo, hi, sd["test"])
    ood, labels = [], []
    for box in cfg["ood_boxes"]:
        pts = sobol(cfg["n_test_ood_per_box"], box["lo"], box["hi"], sd["test_ood"] + len(ood))
        ood.append(pts)
        labels += [box["label"]] * len(pts)
    test_ood = np.vstack(ood)
    rng = np.random.default_rng

    def heights(seed, n, lo_h, hi_h):
        return rng(seed).uniform(lo_h, hi_h, (n, 2)).round(4).tolist()

    sv = heights(sd["scans_validation"], cfg["n_scans_validation"], lo[1:], hi[1:])
    st = heights(sd["scans_test"], cfg["n_scans_test"], lo[1:], hi[1:])
    so = [h for box in cfg["ood_boxes"] for h in heights(sd["scans_ood"] + len(box["label"]), cfg["n_scans_ood_per_box"], box["lo"][1:], box["hi"][1:])]
    manifest = {
        "rules": __doc__,
        "seeds": sd,
        "train_box": [lo.tolist(), hi.tolist()],
        "ood_boxes": cfg["ood_boxes"],
        "sizes": {"candidates": len(cand), "validation": len(val), "test": len(test), "test_ood": len(test_ood)},
        "sha256": {"candidates": _hash(cand), "validation": _hash(val), "test": _hash(test), "test_ood": _hash(test_ood)},
        "scan_heights": {"validation": sv, "test": st, "ood": so},
        "baseline_design": cfg["baseline_design"],
    }
    return Splits(cand, val, test, test_ood, labels, sv, st, so, manifest)


def check_disjoint(*sets: np.ndarray, tol: float = 1e-12) -> bool:
    """No parameter point appears in two sets."""
    rows = [set(map(tuple, np.round(s / tol).astype(np.int64))) for s in sets]
    return all(not (rows[i] & rows[j]) for i in range(len(rows)) for j in range(i + 1, len(rows)))


def save_manifest(path, splits: Splits) -> None:
    path.write_text(json.dumps(splits.manifest, indent=1))
