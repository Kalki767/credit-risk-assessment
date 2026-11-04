"""Tests: contract parity, input validation, threshold logic, reason codes, contract-version refusal, leakage guard.
Run from the repo root:  python -m pytest -q"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import REASON_TEXT, artifacts, predict, to_frame   # noqa: E402
from src.scoring import load_artifacts                              # noqa: E402
from src.validation import check_no_forbidden                       # noqa: E402


def example(name):
    with open(ROOT / "examples" / f"{name}.json") as f:
        return json.load(f)


# 1. parity: the saved transform reproduces the 50 reference rows (1e-6)
def test_parity_on_reference_rows():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_parity.py")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr


# 2. input validation: bad input gives a clear error, never a PD
def test_rejects_out_of_range_value():
    with pytest.raises(ValueError, match="loan_amnt"):
        predict({**example("low_risk"), "loan_amnt": 500000})


def test_rejects_missing_required_field():
    with pytest.raises(ValueError, match="missing required"):
        predict({**example("low_risk"), "loan_amnt": None})


def test_rejects_text_in_numeric_field():
    with pytest.raises(ValueError):
        predict({**example("low_risk"), "annual_inc": "a lot"})


# 3. threshold logic
def test_threshold_logic():
    model = artifacts()[0]
    eps = 1e-9
    got = model.decide([0.0, model.t1 - eps, model.t1, model.t2 - eps, model.t2, 1.0])
    assert list(got) == ["auto-approve", "auto-approve", "review", "review", "auto-decline", "auto-decline"]
    assert model.t1 < 0.125 < model.t2                        # bands sit around the break-even PD


def test_examples_land_in_expected_bands():
    assert predict(example("low_risk"))["decision"] == "auto-approve"
    assert predict(example("borderline"))["decision"] == "review"
    assert predict(example("high_risk"))["decision"] == "auto-decline"


def test_review_trigger_overrides_auto_approve():
    spec = artifacts()[3]
    field = next((f for f in spec["fields"] if f["name"] == "application_type"), {})
    joint = [v for v in field.get("allowed_values", []) if "joint" in str(v).lower()]
    if not joint:
        pytest.skip("data_spec has no joint application level")
    out = predict({**example("low_risk"), "application_type": joint[0]})
    assert out["decision"] == "review" and "joint application" in out["review_triggers"]


# 4. reason codes: present, plain words, ordered, and consistent with the model
def test_reason_codes_plain_and_ordered():
    out = predict(example("high_risk"))
    assert 1 <= len(out["reason_codes"]) <= 4
    assert all(r in REASON_TEXT.values() for r in out["reason_codes"])
    values = list(out["reason_detail"].values())
    assert all(v > 0 for v in values) and values == sorted(values, reverse=True)


def test_reason_codes_match_the_model():
    """Sum over all features of the risk above their best bin = model log-odds - best possible log-odds."""
    model, _, tf, spec = artifacts()
    woe = tf.woe(tf.transform(to_frame(example("high_risk"), spec)))
    raw, row = model.raw_pd(woe)[0], woe.iloc[0]
    best = {c: max(tf.woe_bins_[c]["woe"].values()) for c in model.features}
    best_logit = model.lr_.intercept_[0] + sum(model.coef_[c] * best[c] for c in model.features)
    extra = sum(model.coef_[c] * (row[f"woe_{c}"] - best[c]) for c in model.features)
    assert abs((logit(raw) - best_logit) - extra) < 1e-6


# 5. refuse a model whose contract version differs from the transform's
def test_refuses_mismatched_contract_version(tmp_path):
    models = tmp_path / "models"
    shutil.copytree(ROOT / "models", models)
    with open(models / "metadata.json") as f:
        meta = json.load(f)
    meta["contract_version"] = "9.9.9"
    with open(models / "metadata.json", "w") as f:
        json.dump(meta, f)
    with pytest.raises(ValueError, match="contract_version mismatch"):
        load_artifacts(models_dir=str(models))


# 6. leakage guard: post-issue columns are refused and never used
def test_forbidden_columns_are_refused():
    with pytest.raises(ValueError, match="forbidden"):
        check_no_forbidden(pd.DataFrame({"loan_amnt": [1000], "recoveries": [0.0]}), ["loan_amnt"], ["recoveries"])


def test_model_uses_no_forbidden_or_old_model_columns():
    with open(ROOT / "configs" / "train.yaml") as f:
        cfg = yaml.safe_load(f)
    features = set(artifacts()[0].features)
    assert not features & set(cfg["forbidden_columns"])
    assert not features & {"grade", "sub_grade", "int_rate", "installment"}