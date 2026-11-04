"""Score one application.  CLI: python -m src.predict applicant.json
Input: one raw application + its bureau record as JSON (flat, or {"application": {...}, "bureau": {...}}).
Output: pd, decision, model_version, reason_codes (+ review triggers and model metadata)."""
import io
import json
import sys
import warnings
from functools import lru_cache
from scipy.special import expit

import pandas as pd

from .scoring import load_artifacts
from .validation import validate_frame

MAX_REASONS = 4
warnings.filterwarnings("ignore", message=".*force_all_finite.*")     # harmless library warning
TEXT_FIELDS = {"issue_d", "term", "grade", "sub_grade", "purpose", "title", "application_type", "emp_title",
               "emp_length", "home_ownership", "verification_status", "zip_code", "addr_state", "earliest_cr_line"}
FICO_MIN_TRAINING = 662
REASON_TEXT = {
    "term_months": "Long repayment term (60 months)",
    "fico_mid": "Credit score (FICO) is lower than for our lowest-risk applicants",
    "loan_to_income": "Loan amount is large compared with income",
    "acc_open_past_24mths": "Many credit accounts opened in the last 2 years",
    "dti_cap": "Existing debt payments are high compared with income",
    "num_tl_op_past_12m": "Several new credit accounts opened in the last 12 months",
    "verification_grp": "Income verification status of the application",
    "avg_cur_bal_log": "Low average balance on existing credit accounts",
    "bc_util_cap": "High use of available credit card limits",
    "annual_inc_log": "Income is lower than for our lowest-risk applicants",
    "inq_last_6mths": "Recent credit inquiries in the last 6 months",
    "home_ownership": "Housing status (no mortgage or owned home)",
    "total_rev_hi_lim_log": "Low total credit card limits",
    "purpose_grp": "Loan purpose is associated with higher risk",
}


@lru_cache(maxsize=1)
def artifacts():
    """Load model, metadata, transform and spec once (version-checked)."""
    return load_artifacts()


def to_frame(application, spec):
    """JSON dict -> one-row DataFrame with the same types as a row read from the CSV files."""
    if "application" in application:
        record = {**application.get("application", {}), **application.get("bureau", {})}
    else:
        record = dict(application)
    missing = [f["name"] for f in spec["fields"]
               if f.get("missing_strategy") == "required" and record.get(f["name"]) in (None, "")]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    row = {f["name"]: record.get(f["name"]) for f in spec["fields"]}
    row.update({k: v for k, v in record.items() if k not in row})
    df = pd.read_csv(io.StringIO(pd.DataFrame([row]).to_csv(index=False)))
    text = TEXT_FIELDS | {f["name"] for f in spec["fields"] if f["type"] not in ("float", "int")}
    for c in text & set(df.columns):                       # text stays text even when missing (one-row input)
        df[c] = df[c].map(lambda v: v if pd.isna(v) else str(v)).astype(object)
    return df

def reason_codes(model, tf, woe_row):
    """Features costing the most risk vs their best bin: weight x (WoE - best WoE) > 0, largest first.
    Also the lowest possible PD (every feature in its best bin): the reference point for the reasons."""
    extra, best_logit = {}, float(model.lr_.intercept_[0])
    for c in model.features:
        best = max(tf.woe_bins_[c]["woe"].values())
        best_logit += model.coef_[c] * best
        extra[c] = float(model.coef_[c] * (woe_row[f"woe_{c}"] - best))
    top = sorted(((v, c) for c, v in extra.items() if v > 1e-6), reverse=True)[:MAX_REASONS]
    pd_floor = float(model.recalibrate(expit(best_logit)))
    return [REASON_TEXT[c] for _, c in top], {c: round(v, 4) for v, c in top}, pd_floor

def review_triggers(row):
    """Rules that send an auto-approval to manual review (DESIGN.md)."""
    found = []
    if "joint" in str(row.get("application_type", "")).lower():
        found.append("joint application")
    if pd.notna(row.get("fico_mid")) and row["fico_mid"] < FICO_MIN_TRAINING:
        found.append("FICO below the training range")
    if pd.notna(row.get("dti")) and pd.notna(row.get("dti_cap")) and row["dti"] > row["dti_cap"]:
        found.append("debt-to-income above the training range")
    return found


def predict(application):
    """One raw application (+ bureau record) -> pd, decision, model_version, reason_codes."""
    model, meta, tf, spec = artifacts()
    raw = to_frame(application, spec)
    try:
        clean = tf.transform(raw)
    except Exception as e:                                   # e.g. text where a number is expected
        raise ValueError(f"could not process the application: {e}") from e
    problems = validate_frame(clean, spec)
    if problems:
        raise ValueError("invalid application:\n- " + "\n- ".join(problems))
    woe = tf.woe(clean)
    p = float(model.predict_pd(woe)[0])
    decision = str(model.decide([p])[0])
    triggers = review_triggers(clean.iloc[0])
    if decision == "auto-approve" and triggers:
        decision = "review"
    reasons, detail, pd_floor = reason_codes(model, tf, woe.iloc[0])
    meaning = (f"Approved. These factors raise the PD above the lowest possible ({pd_floor:.1%}), "
               f"but not above the approval line." if decision == "auto-approve"
               else "Main factors behind the higher risk, largest first.")
    return {"pd": round(p, 4), "decision": decision, "model_version": meta["model_version"],
            "reason_codes": reasons, "reason_codes_meaning": meaning,
            "pd_lowest_possible": round(pd_floor, 4), "review_triggers": triggers, "reason_detail": detail,
            "metadata": {"training_date": meta["training_date"], "git_commit": meta["git_commit"],
                         "contract_version": meta["contract_version"], "thresholds": meta["thresholds"]}}

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m src.predict applicant.json")
        return 2
    with open(argv[0]) as f:
        application = json.load(f)
    try:
        print(json.dumps(predict(application), indent=2))
        return 0
    except ValueError as e:
        print(json.dumps({"error": str(e)}, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())