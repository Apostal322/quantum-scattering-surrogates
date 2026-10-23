"""Run one Phase 7 experiment: python scripts/phase7/run.py <name>   (configs/phase7/<name>.json)

Writes results/phase7/<name>/results.json with provenance (config, git commit, clean-tree flag).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qscatter.phase6.experiments import provenance  # noqa: E402
from qscatter.phase7 import basis, mlp_diagnostic, robustness, slice1d  # noqa: E402
from qscatter.phase7 import experiments as ex  # noqa: E402
from qscatter.phase7.validation import run_validation  # noqa: E402

RUNNERS = {"validation": run_validation, **ex.RUNNERS, **slice1d.RUNNERS, **basis.RUNNERS, **robustness.RUNNERS, **mlp_diagnostic.RUNNERS}

if __name__ == "__main__":
    name = sys.argv[1]
    cfg = json.loads((REPO / "configs" / "phase7" / f"{name}.json").read_text())
    t0 = time.perf_counter()
    data = RUNNERS[cfg.get("runner", name)](cfg)
    data["provenance"] = provenance(cfg)
    data["wall_seconds"] = time.perf_counter() - t0
    out = REPO / "results" / "phase7" / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(data, indent=1, default=float))
    print(f"{name}: {data['wall_seconds']:.0f} s -> {out.relative_to(REPO)}/results.json")
