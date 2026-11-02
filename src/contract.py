"""Build contract/data_spec.json from the fitted transform plus the A1-A7 field knowledge."""
import copy

import numpy as np
import pandas as pd

from .features import (CONTRACT_VERSION, FA_COLS, FACTOR_NAMES, FACTOR_SCORE_COLS, MIN_GROUP,
                       MODEL_FEATURES, PCA_COLS, PCA_LOG, PURPOSE_RISK, VERIF_MAP)

HIGH_BUREAU = ("high", "credit-bureau record; balances are exact (0.1% round thousands = chance rate)")

# name, type, min, max, role, cleaning, missing_strategy, (reliability, reason), allowed values
FIELDS = [
    ("loan_amnt", "float", 1000, 40000, "model input",
     "divided by 10 when the installment is exactly 1/10 of the amortization payment (ratio 0.099-0.101); "
     "not capped (bounded by lending policy)", "required",
     ("high", "platform-set; installment formula matches 99.96% of loans"), None),
    ("term", "category", None, None, "model input (as term_months)", "parsed to term_months (36 or 60)",
     "required", ("high", "platform-set"), ["36 months", "60 months"]),
    ("int_rate", "float", 0, 40, "cleaning check only",
     "used to verify installment (x10 rule); excluded from the model: old-model pricing (set from sub_grade)",
     "required", ("high", "platform-set"), None),
    ("installment", "float", 0.01, None, "cleaning check only",
     "used for the x10 rule; excluded from the model: encodes int_rate (Spearman 0.999 within term)",
     "required", ("high", "platform-set; 11 mismatches flagged"), None),
    ("grade", "category", None, None, "excluded", "old model output; benchmark only", "n/a",
     ("high", "platform-set"), list("ABCDEFG")),
    ("sub_grade", "category", None, None, "excluded", "old model output; benchmark only", "n/a",
     ("high", "platform-set"), None),
    ("emp_title", "text", None, None, "excluded",
     "definition changed in 2013Q4 (employer names -> job titles; PSI 2013 vs 2014 = 0.852); fairness proxy risk",
     "n/a", ("low", "free text, 13,327 distinct spellings; meaning changed over time"), None),
    ("emp_length", "category", None, None, "excluded",
     "no predictive value on new loans (2015 AUC 0.5076 with 2013-14 WoE)", "n/a",
     ("low", "self-reported; 5% blank, same people as blank emp_title"), None),
    ("home_ownership", "category", None, None, "model input",
     "none (3 levels, all >= 1,406 loans); unseen level -> WoE 0", "required",
     ("medium", "self-reported but consistent: mortgage holders carry mortgage-sized balances"),
     ["MORTGAGE", "RENT", "OWN"]),
    ("annual_inc", "float", 1000, 10_000_000, "model input (as annual_inc_log, loan_to_income)",
     "values below 1,000 treated as thousands (x1,000)", "required",
     ("low", "self-reported; 84% round thousands; verification does not change rounding; "
             "implausible extremes ($7.5M 'Verified')"), None),
    ("verification_status", "category", None, None, "model input (as verification_grp)",
     "Verified and Source Verified merged into 'verified'", "required",
     ("medium", "accurate label but confounded: the platform verified larger, riskier loans"),
     ["Verified", "Source Verified", "Not Verified"]),
    ("issue_d", "date", None, None, "time split; history length",
     "three formats parsed explicitly to month precision", "required", ("high", "platform-set"), None),
    ("purpose", "category", None, None, "model input (as purpose_grp)",
     f"levels with < {MIN_GROUP} loans (<= 2014) and unseen levels -> 'other'; scorecard uses 3 risk levels",
     "required", ("medium", "self-declared by the applicant"), None),
    ("title", "text", None, None, "excluded", "redundant with purpose (63% identical)", "n/a",
     ("low", "free text"), None),
    ("zip_code", "text", None, None, "excluded", "location proxy risk; high cardinality", "n/a",
     ("medium", "format valid; 27 ZIP/state mismatches flagged"), None),
    ("addr_state", "category", None, None, "excluded",
     "names/case cleaned to codes; excluded: location proxy risk, 2015 AUC 0.512", "n/a",
     ("medium", "cleaned to two-letter codes"), None),
    ("dti", "float", 0, None, "model input (as dti_cap)", "", "required",
     ("medium", "computed by the platform from self-reported income"), None),
    ("application_type", "category", None, None, "excluded",
     "constant in training (Individual only); joint applications -> manual review", "n/a",
     ("high", "platform-set"), ["Individual", "Joint App"]),
    ("bureau_recovery_flag", "int", None, None, "excluded",
     "LEAKAGE: post-default recovery status (100% default when 1); removed by cleaning", "n/a",
     ("n/a", "not known at application time"), None),
    ("acc_open_past_24mths", "int", 0, None, "model input", "log1p for PCA/FA only", "required", HIGH_BUREAU, None),
    ("avg_cur_bal", "float", 0, None, "model input (as avg_cur_bal_log)", "", "required",
     ("high", "bureau record; near-redundant: balance / number of accounts (VIF 168)"), None),
    ("bc_util", "float", 0, None, "model input (as bc_util_cap)", "negative values: sign flipped (absolute value)",
     "missing -> flag bc_util_missing + median (fitted <= 2014)", ("high", "bureau record"), None),
    ("delinq_2yrs", "int", 0, None, "model input", "none", "required", HIGH_BUREAU, None),
    ("earliest_cr_line", "date", None, None, "model input (as credit_hist_months)",
     "three formats parsed; dates after issue_d set to missing + flag",
     "missing -> median history of its total_acc group (fitted <= 2014), at least months since last delinquency",
     ("high", "bureau record; 1% impossible dates"), None),
    ("fico_range_low", "int", 300, 850, "model input (as fico_mid)", "swapped with high when high < low",
     "required", ("high", "bureau score"), None),
    ("fico_range_high", "int", 300, 850, "model input (as fico_mid)", "swapped with low when high < low",
     "required", ("high", "bureau score"), None),
    ("inq_last_6mths", "int", 0, None, "model input", "log1p for PCA/FA only", "required", HIGH_BUREAU, None),
    ("mort_acc", "int", 0, None, "model input", "log1p for PCA/FA only", "required", HIGH_BUREAU, None),
    ("mths_since_last_delinq", "float", 0, None, "model input",
     "none", "missing = never delinquent -> flag ever_delinq; kept missing (own WoE bin)",
     HIGH_BUREAU, None),
    ("num_tl_op_past_12m", "int", 0, None, "model input", "log1p for PCA/FA only", "required", HIGH_BUREAU, None),
    ("open_acc", "int", 0, None, "model input", "log1p for PCA/FA only", "required", HIGH_BUREAU, None),
    ("pub_rec", "int", 0, None, "model input", "none", "required", HIGH_BUREAU, None),
    ("pub_rec_bankruptcies", "int", 0, None, "model input", "none", "required", HIGH_BUREAU, None),
    ("revol_bal", "float", 0, None, "model input (as revol_bal_log)", "", "required", HIGH_BUREAU, None),
    ("revol_util", "float", 0, None, "model input (as revol_util_cap)",
     "negative values: sign flipped (absolute value)",
     "missing -> recomputed as revol_bal / total_rev_hi_lim x 100 (0 when there is no revolving credit)",
     ("high", "recomputation from balance / limit agrees within 5 pts for 96.2%"), None),
    ("tot_cur_bal", "float", 0, None, "model input (as tot_cur_bal_log)", "", "required", HIGH_BUREAU, None),
    ("tot_hi_cred_lim", "float", 0, None, "model input (as tot_hi_cred_lim_log)", "", "required", HIGH_BUREAU, None),
    ("total_acc", "int", 0, None, "model input", "none", "required", HIGH_BUREAU, None),
    ("total_rev_hi_lim", "float", 0, None, "model input (as total_rev_hi_lim_log)", "", "required",
     HIGH_BUREAU, None),
]


def _cap_text(rule):
    how = "p99.5" if rule["method"] == "p99.5" else "IQR fence"
    text = f"capped at {how} = {rule['upper']:,.2f} (fitted <= 2014)"
    return text + "; then log1p" if rule["log1p"] else text


def build_data_spec(tf, fit_out, exclude):
    """tf: fitted CreditRiskTransform; fit_out: tf.transform() of the <= 2014 rows; exclude: reasons from a7_spec."""
    p = tf.params_
    fields = []
    for name, typ, lo, hi, role, cleaning, missing, (rel, reason), allowed in FIELDS:
        if name in p["caps"]:
            cleaning = "; ".join(t for t in [cleaning, _cap_text(p["caps"][name])] if t)
        f = {"name": name, "type": typ, "min": lo, "max": hi, "role": role, "cleaning": cleaning or "none",
             "missing_strategy": missing, "reliability": rel, "reliability_reason": reason}
        if allowed:
            f["allowed_values"] = allowed
        fields.append(f)

    fa = tf.models_["fa"]
    load = pd.DataFrame(fa.loadings_, index=FA_COLS, columns=FACTOR_NAMES)
    shares = fa.get_factor_variance()[1]
    factors = []
    for i, name in enumerate(FACTOR_NAMES):
        s = load[name][load[name].abs() >= 0.4].sort_values(key=abs, ascending=False)
        factors.append({"name": name, "score_column": FACTOR_SCORE_COLS[i],
                        "method": "factor_analysis_promax (minres, 4 factors)",
                        "top_loadings": {k: round(float(v), 3) for k, v in s.items()},
                        "variance_explained": round(float(shares[i]), 3)})

    pca = tf.models_["pca"]
    comp = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), index=PCA_COLS,
                        columns=[f"pc{i}" for i in range(1, pca.n_components_ + 1)])
    return {
        "contract_version": CONTRACT_VERSION,
        "fit_cutoff": str(tf.fit_cutoff.date()),
        "n_fit_rows": int(tf.n_fit_rows_),
        "input": "one row per application with its bureau record (the holdout_applications.csv columns); "
                 "validation ranges apply to cleaned values",
        "fields": fields,
        "model_features": MODEL_FEATURES,
        "excluded": exclude,
        "factors": factors,
        "factor_notes": "promax lets factors correlate, so per-factor variance shares overlap; "
                        "factor score correlations (<= 2014): "
                        + str(fit_out[FACTOR_SCORE_COLS].corr().round(2).to_dict()),
        "pca": {"inputs": PCA_COLS, "log1p_inputs": PCA_LOG,
                "variance_explained": [round(float(v), 3) for v in pca.explained_variance_ratio_],
                "top_loadings": {c: {k: round(float(v), 3) for k, v in
                                     comp[c][comp[c].abs() >= 0.4].sort_values(key=abs, ascending=False).items()}
                                 for c in comp.columns}},
        "fa_inputs": FA_COLS,
        "caps": p["caps"],
        "imputation": p["impute"],
        "feature_caps": p["feature_caps"],
        "category_groupings": {"min_group_size": MIN_GROUP, "purpose_keep": p["purpose_keep"],
                               "purpose_risk_levels_scorecard": PURPOSE_RISK, "verification_map": VERIF_MAP,
                               "unseen_level": "purpose -> 'other'; any WoE level -> 0 (neutral)"},
        "woe_bins": tf.woe_bins_,
    }