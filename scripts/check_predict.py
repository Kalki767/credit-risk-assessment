"""Check that predict.py (one application at a time) gives the same PD as the batch holdout scoring."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict   # noqa: E402

raw = pd.read_csv(ROOT / "data/raw/holdout_applications.csv").head(50)
batch = pd.read_csv(ROOT / "reports/holdout_scores.csv").set_index("id")["pd"]
diffs, out = [], None
for _, r in raw.iterrows():
    out = predict({k: (None if pd.isna(v) else v) for k, v in r.items()})
    diffs.append(abs(out["pd"] - batch[r["id"]]))
print("rows checked:", len(diffs), "| max |single - batch| PD difference:", round(max(diffs), 6))
print(json.dumps(out, indent=2))
print({k: r[k] for k in ["issue_d", "term", "int_rate", "installment", "emp_length", "earliest_cr_line",
                         "revol_util", "application_type", "verification_status"]})