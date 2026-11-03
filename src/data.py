"""Load the config and the contract, split the data in time, and build the cross-validation folds."""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]          # project root, wherever the code is run from


def load_config(path="configs/train.yaml"):
    with open(ROOT / path) as f:
        return yaml.safe_load(f)


def file_hash(path):
    """SHA-256 fingerprint of a file: any change to the data gives a different value."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_contract(cfg):
    """The fitted transform, the data spec and the analytical dataset; refuses mismatched versions."""
    cdir = ROOT / cfg["paths"]["contract_dir"]
    tf = joblib.load(cdir / "transform.joblib")
    spec = json.load(open(cdir / "data_spec.json"))
    if tf.contract_version != spec["contract_version"]:
        raise ValueError(f"contract_version mismatch: transform {tf.contract_version} vs spec {spec['contract_version']}")
    data = pd.read_parquet(cdir / "analytical_dataset.parquet")
    return tf, spec, data


def time_split(df, cfg):
    """Train = issued <= fit_cutoff; validation = issued within the validation window."""
    d = cfg["data"]
    train = df[df["issue_dt"] <= pd.Timestamp(d["fit_cutoff"])]
    valid = df[df["issue_dt"].between(pd.Timestamp(d["valid_start"]), pd.Timestamp(d["valid_end"]))]
    return train, valid


def cv_folds(train, cfg):
    """Expanding-window folds: train on everything before a quarter, validate on that quarter."""
    quarter = train["issue_dt"].dt.to_period("Q")
    folds = []
    for q in cfg["data"]["cv_quarters"]:
        period = pd.Period(q, freq="Q")
        fit_idx = train.index[train["issue_dt"] < period.start_time]
        val_idx = train.index[quarter == period]
        folds.append((q, fit_idx, val_idx))
    return folds