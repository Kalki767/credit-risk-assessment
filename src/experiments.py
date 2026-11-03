"""Append one row per experiment to the CSV experiment log."""
import json
from datetime import datetime

import pandas as pd

from .data import ROOT


def log_experiment(cfg, model, variant, cv_table, features, params, data_hash, contract_version, notes=""):
    """cv_table: per-fold metrics from cv_evaluate (columns period, auc, gini, ks)."""
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "variant": variant,
        "cv_auc_mean": round(cv_table["auc"].mean(), 4),
        "cv_ks_mean": round(cv_table["ks"].mean(), 4),
        "cv_auc_by_fold": json.dumps(dict(zip(cv_table["period"], cv_table["auc"].round(4)))),
        "n_features": len(features),
        "features": json.dumps(features),
        "params": json.dumps(params),
        "seed": cfg["seed"],
        "data_hash": data_hash[:16],
        "contract_version": contract_version,
        "notes": notes,
    }
    path = ROOT / cfg["paths"]["experiment_log"]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)
    return row