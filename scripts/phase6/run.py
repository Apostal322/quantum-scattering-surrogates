"""Run one Phase 6 experiment from its config: python scripts/phase6/run.py rb_1d|rb_3d|nwidth

Writes results/phase6/<name>/results.json (with provenance: config, git commit, versions).
Tables and figures are produced separately by scripts/phase6/analyze.py.
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # deterministic timing, no BLAS oversubscription
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qscatter.phase6 import experiments as ex  # noqa: E402

RUNNERS = {"rb_1d": ex.run_rb_1d, "rb_3d": ex.run_rb_3d, "nwidth": ex.run_nwidth}

if __name__ == "__main__":
    name = sys.argv[1]
    cfg = json.loads((REPO / "configs" / "phase6" / f"{name}.json").read_text())
    t0 = time.perf_counter()
    data = RUNNERS[name](cfg)
    data["wall_seconds"] = time.perf_counter() - t0
    out = REPO / "results" / "phase6" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, indent=1, default=float))
    print(f"{name}: {data['wall_seconds']:.0f} s -> {out.relative_to(REPO)}/results.json")
