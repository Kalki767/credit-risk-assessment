"""Score the 2016 holdout and measure drift vs the 2015 validation year.
Run: python scripts/score_holdout.py   ->   reports/holdout_scores.csv (id, pd), reports/drift_holdout.csv"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import load_config, load_contract, time_split   # noqa: E402
from src.metrics import psi                                    # noqa: E402
from src.scoring import load_artifacts, score_frame            # noqa: E402
from src.validation import check_no_forbidden, validate_frame  # noqa: E402

ALERTS = [(0.10, "OK"), (0.25, "WARN: investigate"), (np.inf, "ACT: review / recalibrate / retrain")]


def alert(value):
    return next(label for limit, label in ALERTS if value < limit)


def main(config_path="configs/train.yaml"):
    cfg = load_config(config_path)
    model, meta, tf, spec = load_artifacts(cfg["paths"]["models_dir"], cfg["paths"]["contract_dir"])

    # 1. score the holdout through the shared path
    raw = pd.read_csv(ROOT / cfg["paths"]["holdout_raw"])
    hold = score_frame(raw, model, tf)
    check_no_forbidden(hold, model.features, cfg["forbidden_columns"])
    assert len(hold) == len(raw), "every holdout row must get a PD"
    problems = validate_frame(hold, spec)
    print(f"holdout rows {len(hold)} | validation problems: {len(problems)}")
    for p in problems:
        print("  -", p)
    ids = hold["id"] if "id" in hold.columns else raw.loc[hold.index, "id"]
    out = ROOT / cfg["paths"]["reports_dir"]
    pd.DataFrame({"id": ids.values, "pd": hold["pd"].values}).to_csv(out / "holdout_scores.csv", index=False)

    # 2. the 2015 validation year, scored the same way (production PDs)
    _, _, data = load_contract(cfg)
    _, valid = time_split(data, cfg)
    pd_valid = model.predict_pd(tf.woe(valid))

    # 3. PSI: PD distribution + top 10 model inputs by IV
    top10 = sorted(model.features, key=lambda c: spec["woe_bins"][c]["iv"], reverse=True)[:10]
    rows = [{"variable": "pd", "psi": psi(pd_valid, hold["pd"])}]
    rows += [{"variable": c, "psi": psi(valid[c], hold[c])} for c in top10]
    drift = pd.DataFrame(rows)
    drift["alert"] = drift["psi"].apply(alert)
    drift.to_csv(out / "drift_holdout.csv", index=False)
    print(drift.round(4).to_string(index=False))

    # 4. decision mix and PD level, 2015 vs holdout
    mix = pd.DataFrame({"2015": pd.Series(model.decide(pd_valid)).value_counts(normalize=True),
                        "holdout 2016": hold["decision"].value_counts(normalize=True)}).round(3)
    print(mix)
    print(f"mean PD: 2015 {pd_valid.mean():.4f} | holdout {hold['pd'].mean():.4f}")


if __name__ == "__main__":
    main()