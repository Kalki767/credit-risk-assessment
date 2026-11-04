"""One command: python scripts/train.py [--config configs/train.yaml]
Trains the WoE scorecard from the contract, evaluates it, sets thresholds, logs the run, saves model + metadata."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import cv_folds, file_hash, load_config, load_contract, time_split   # noqa: E402
from src.experiments import log_experiment                                         # noqa: E402
from src.metrics import probability_metrics, ranking_metrics                       # noqa: E402
from src.model import ScorecardModel, fold_woe                                     # noqa: E402
from src.policy import band_thresholds, best_cutoff, policy_result                 # noqa: E402
from src.validation import check_no_forbidden, validate_or_fail                    # noqa: E402

GRADE_NUM = {g: i for i, g in enumerate("ABCDEFG", start=1)}


def git_info():
    """Current commit and whether the working tree has uncommitted changes (lineage)."""
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True


def cross_validate(train, cfg, features, rules, target, C):
    """Fold-honest expanding-window CV: per-fold metrics and out-of-fold PDs."""
    rows, oof = [], pd.Series(np.nan, index=train.index)
    for q, fit_idx, val_idx in cv_folds(train, cfg):
        f, v = train.loc[fit_idx], train.loc[val_idx]
        m = ScorecardModel(features, C).fit(fold_woe(f, f, features, rules, target), f[target])
        oof.loc[val_idx] = m.raw_pd(fold_woe(f, v, features, rules, target))
        rows.append({"period": q, **ranking_metrics(v[target], oof.loc[val_idx])})
    return pd.DataFrame(rows), oof


def main(config_path):
    cfg = load_config(config_path)
    np.random.seed(cfg["seed"])
    d, mc, pc, econ = cfg["data"], cfg["model"], cfg["policy"], cfg["economics"]
    target, gain, loss = d["target"], econ["gain_if_repaid"], econ["loss_if_default"]

    # 1. contract (versions checked), forbidden columns, input validation: fail fast
    tf, spec, data = load_contract(cfg)
    data_hash = file_hash(ROOT / cfg["paths"]["contract_dir"] / "analytical_dataset.parquet")
    check_no_forbidden(data, mc["features"], cfg["forbidden_columns"])
    validate_or_fail(data, spec)
    train, valid = time_split(data, cfg)
    woe = tf.woe(data)                                     # transform.joblib used unchanged
    print(f"contract {spec['contract_version']} | train {len(train)} | valid {len(valid)} | data {data_hash[:12]}")

    # 2. time-aware CV with fold-honest WoE
    cv_table, oof = cross_validate(train, cfg, mc["features"], spec["woe_bins"], target, mc["C"])
    print(f"CV AUC {cv_table['auc'].mean():.4f} | by quarter {cv_table['auc'].round(4).tolist()}")

    # 3. final model on all loans <= fit cutoff, evaluated once on the validation year
    model = ScorecardModel(mc["features"], mc["C"], mc["version"]).fit(woe.loc[train.index], train[target])
    pd_raw = model.raw_pd(woe.loc[valid.index])
    yv, amt = valid[target].values, valid["loan_amnt"].values
    valid_metrics = {k: float(v) for k, v in {**ranking_metrics(yv, pd_raw), **probability_metrics(yv, pd_raw)}.items()}
    grade_auc = float(ranking_metrics(yv, valid["grade"].map(GRADE_NUM))["auc"])
    print(f"validation AUC {valid_metrics['auc']:.4f} (grade alone {grade_auc:.4f}) | "
          f"mean PD {valid_metrics['mean_pd']:.4f} vs actual {valid_metrics['default_rate']:.4f}")

    # 4. evaluation policy: chosen on 2014 out-of-fold PDs, tested on the validation year
    m = oof.notna()
    cut = best_cutoff(oof[m], train.loc[m, target], train.loc[m, "loan_amnt"], gain, loss, pc["cutoff_grid"])
    t1_eval, t2_eval = band_thresholds(oof[m], cut, pc["review_share"])
    eval_approve = policy_result(pd_raw < t1_eval, yv, amt, gain, loss)
    print(f"evaluation: cut-off {cut:.3f}, bands {t1_eval:.3f} / {t2_eval:.3f} | auto-approve on validation: "
          f"{ {k: round(v, 3) for k, v in eval_approve.items()} }")

    # 5. production refresh (B17): recalibration on the validation year, bands centred on the break-even
    if pc["recalibrate_on_validation"]:
        model.fit_recalibration(pd_raw, yv)
    break_even = gain / (gain + loss)
    model.set_thresholds(*band_thresholds(model.recalibrate(pd_raw), break_even, pc["review_share"]))
    print(f"recalibration slope {model.cal_slope:.3f}, intercept {model.cal_intercept:.3f} | "
          f"production bands {model.t1:.3f} / {model.t2:.3f}")

    # 6. save model + metadata, log the run
    commit, dirty = git_info()
    meta = {
        "model_version": mc["version"],
        "model_type": "WoE scorecard (logistic regression on contract WoE) + Platt recalibration + 3 decision bands",
        "training_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit, "git_dirty": dirty,
        "data_hash_sha256": data_hash,
        "contract_version": spec["contract_version"], "fit_cutoff": str(spec["fit_cutoff"]),
        "train_rows": int(len(train)), "valid_rows": int(len(valid)),
        "valid_window": [str(d["valid_start"]), str(d["valid_end"])],
        "seed": cfg["seed"], "features": mc["features"], "C": mc["C"],
        "coefficients": {k: float(v) for k, v in model.coef_.items()}, "intercept": float(model.lr_.intercept_[0]),
        "recalibration": {"slope": model.cal_slope, "intercept": model.cal_intercept,
                          "fitted_on": "validation year, after evaluation (production refresh)"},
        "thresholds": {"auto_approve_below": model.t1, "auto_decline_from": model.t2,
                       "break_even_pd": break_even, "review_share": pc["review_share"]},
        "evaluation_thresholds": {"cutoff": cut, "auto_approve_below": t1_eval, "auto_decline_from": t2_eval,
                                  "chosen_on": "2014 out-of-fold PDs"},
        "metrics": {"cv_auc_mean": float(cv_table["auc"].mean()),
                    "cv_auc_by_quarter": {q: float(a) for q, a in zip(cv_table["period"], cv_table["auc"])},
                    "validation": valid_metrics, "validation_grade_auc": grade_auc,
                    "validation_auto_approve_eval_bands": eval_approve},
    }
    out = ROOT / cfg["paths"]["models_dir"]
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "scorecard.joblib")
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log_experiment(cfg, "woe_scorecard", "train.py run", cv_table, mc["features"], {"C": mc["C"]}, data_hash,
                   spec["contract_version"], notes=f"valid AUC {valid_metrics['auc']:.4f}; commit {commit[:8]}"
                                                   f"{' (dirty)' if dirty else ''}")
    print(f"saved {out / 'scorecard.joblib'} and metadata.json (commit {commit[:8]}{', DIRTY' if dirty else ''})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the PD scorecard from the data contract.")
    ap.add_argument("--config", default="configs/train.yaml")
    main(ap.parse_args().config)