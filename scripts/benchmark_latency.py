"""Benchmark single-application scoring latency (predict() end to end) -> reports/latency.json"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict   # noqa: E402

apps = []
for name in ("low_risk", "borderline", "high_risk"):
    with open(ROOT / "examples" / f"{name}.json") as f:
        apps.append(json.load(f))
for a in apps * 10:                        # warm-up: loads the model and transform once
    predict(a)
times = []
for i in range(600):
    t0 = time.perf_counter()
    predict(apps[i % 3])
    times.append((time.perf_counter() - t0) * 1000)
result = {"n_calls": len(times), "p50_ms": round(float(np.percentile(times, 50)), 2),
          "p95_ms": round(float(np.percentile(times, 95)), 2), "max_ms": round(float(max(times)), 2),
          "target_ms": 100, "machine": f"{platform.machine()} {platform.processor()}".strip(),
          "python": platform.python_version()}
with open(ROOT / "reports" / "latency.json", "w") as f:
    json.dump(result, f, indent=2)
print(result)