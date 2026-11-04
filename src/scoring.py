"""Load the saved model + contract, and score raw applications. Shared by score_holdout.py, predict.py and the Space."""
import json

import joblib

from .data import ROOT


def load_artifacts(models_dir="models", contract_dir="contract"):
    """Model, metadata, transform and spec; refuses to run if their contract versions differ."""
    model = joblib.load(ROOT / models_dir / "scorecard.joblib")
    meta = json.load(open(ROOT / models_dir / "metadata.json"))
    tf = joblib.load(ROOT / contract_dir / "transform.joblib")
    spec = json.load(open(ROOT / contract_dir / "data_spec.json"))
    versions = {"model": meta["contract_version"], "transform": tf.contract_version, "spec": spec["contract_version"]}
    if len(set(versions.values())) != 1:
        raise ValueError(f"contract_version mismatch: {versions}")
    return model, meta, tf, spec


def score_frame(raw_df, model, tf):
    """Raw application + bureau rows -> transformed rows with pd and decision (transform used unchanged)."""
    out = tf.transform(raw_df)
    out["pd"] = model.predict_pd(tf.woe(out))
    out["decision"] = model.decide(out["pd"])
    return out