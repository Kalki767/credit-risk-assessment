"""Parity check: transform.joblib must reproduce contract/reference_transforms.csv within 1e-6.

Run from the project root:  python scripts/check_parity.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                         # so the pickled transform can import src/


def check_parity(contract_dir=ROOT / "contract", tol=1e-6):
    tf = joblib.load(contract_dir / "transform.joblib")
    spec = json.load(open(contract_dir / "data_spec.json"))
    if tf.contract_version != spec["contract_version"]:
        raise ValueError(f"contract_version mismatch: transform {tf.contract_version}, spec {spec['contract_version']}")

    ref = pd.read_csv(contract_dir / "reference_transforms.csv", dtype={"raw__id": str}, low_memory=False)
    raw = ref[[c for c in ref.columns if c.startswith("raw__")]].rename(columns=lambda c: c[len("raw__"):])
    out = tf.transform(raw)
    out = out.join(tf.woe(out))

    failures = []
    for col in [c for c in ref.columns if c.startswith("expected__")]:
        name = col[len("expected__"):]
        got, exp = out[name], ref[col]
        if pd.api.types.is_datetime64_any_dtype(got):
            got = got.dt.strftime("%Y-%m-%d")
        if pd.api.types.is_numeric_dtype(exp) and pd.api.types.is_numeric_dtype(got):
            ok = np.isclose(got.astype(float), exp.astype(float), rtol=0, atol=tol, equal_nan=True).all()
        else:
            ok = (got.astype(str).where(got.notna(), "NA") == exp.astype(str).where(exp.notna(), "NA")).all()
        if not ok:
            failures.append(name)

    n = sum(c.startswith("expected__") for c in ref.columns)
    print(f"parity: {n - len(failures)}/{n} columns match on {len(ref)} rows (tolerance {tol})")
    if failures:
        print("FAILED:", failures)
    return not failures


if __name__ == "__main__":
    sys.exit(0 if check_parity() else 1)