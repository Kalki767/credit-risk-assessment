"""A3-A5 feature rules: fit numbers on loans issued <= the fit cutoff, apply them unchanged to any rows."""
import numpy as np
import pandas as pd
from .cleaning import clean

FIT_CUTOFF = pd.Timestamp("2014-12-31")
MIN_GROUP = 100                                              # A5: smallest category kept on its own

LOG_CAP_COLS = ["annual_inc", "revol_bal", "tot_cur_bal", "avg_cur_bal",
                "tot_hi_cred_lim", "total_rev_hi_lim"]       # A3: cap at p99.5, then log1p
FENCE_COLS = ["dti", "revol_util", "bc_util"]                # A3: cap at the IQR fence
VERIF_MAP = {"Verified": "verified", "Source Verified": "verified", "Not Verified": "not_verified"}
PURPOSE_RISK = {"small_business": "high",                    # A5: scorecard coarse classes
                "debt_consolidation": "medium", "other": "medium", "medical": "medium",
                "credit_card": "low", "home_improvement": "low", "major_purchase": "low", "car": "low"}

MODEL_FEATURES = [                                           # final 29 (a7_spec.json)
    "acc_open_past_24mths", "annual_inc_log", "avg_cur_bal_log", "bc_util_cap", "bc_util_missing",
    "credit_hist_months", "delinq_2yrs", "dti_cap", "ever_delinq", "fico_mid", "home_ownership",
    "inq_last_6mths", "loan_amnt", "loan_to_income", "mort_acc", "mths_since_last_delinq",
    "num_tl_op_past_12m", "open_acc", "pub_rec", "pub_rec_bankruptcies", "purpose_grp",
    "revol_bal_log", "revol_util_cap", "term_months", "tot_cur_bal_log", "tot_hi_cred_lim_log",
    "total_acc", "total_rev_hi_lim_log", "verification_grp",
]


def history_months(df):
    """Months between the first credit line and the loan's issue date."""
    return (df["issue_dt"] - df["earliest_cr_dt"]).dt.days / 30.44


def fit_basic(fit_df):
    """Learn every A3-A5 number from cleaned loans issued <= FIT_CUTOFF. Returns plain numbers/lists."""
    caps = {}
    for c in LOG_CAP_COLS:                                                  # A3
        caps[c] = {"method": "p99.5", "upper": float(fit_df[c].quantile(0.995)), "log1p": True}
    for c in FENCE_COLS:
        q1, q3 = fit_df[c].quantile([0.25, 0.75])
        caps[c] = {"method": "iqr_fence", "upper": float(q3 + 1.5 * (q3 - q1)), "log1p": False}

    _, edges = pd.qcut(fit_df["total_acc"], 5, retbins=True)                # A4: history by total_acc group
    edges[0], edges[-1] = -np.inf, np.inf
    groups = pd.cut(fit_df["total_acc"], edges)
    hist_medians = history_months(fit_df).groupby(groups, observed=False).median().round(1).tolist()

    income_capped = fit_df["annual_inc"].clip(upper=caps["annual_inc"]["upper"])   # A4: ratio cap
    lti_cap = float((fit_df["loan_amnt"] / income_capped).quantile(0.995))

    counts = fit_df["purpose"].value_counts()                                # A5
    return {
        "caps": caps,
        "impute": {"total_acc_edges": edges.tolist(), "hist_months_by_group": hist_medians,
                   "bc_util_median": float(fit_df["bc_util"].median())},
        "feature_caps": {"loan_to_income": lti_cap},
        "purpose_keep": sorted(counts[counts >= MIN_GROUP].index.tolist()),
    }


def apply_basic(df, p):
    """Apply the fitted A3-A5 rules to cleaned rows (any period). Never re-learns anything."""
    out = df.copy()

    for c, rule in p["caps"].items():                                        # A3 caps and logs
        out[f"{c}_cap"] = out[c].clip(upper=rule["upper"])
        if rule["log1p"]:
            out[f"{c}_log"] = np.log1p(out[f"{c}_cap"])

    imp = p["impute"]                                                        # A4 imputation
    group = pd.cut(out["total_acc"], imp["total_acc_edges"], labels=False)
    fill = np.fmax(group.map(dict(enumerate(imp["hist_months_by_group"]))), out["mths_since_last_delinq"])
    out["credit_hist_months"] = history_months(out).fillna(fill)

    limit = out["total_rev_hi_lim"]
    recomputed = pd.Series(np.where(limit > 0, out["revol_bal"] / limit * 100, 0.0), index=out.index)
    out["revol_util"] = out["revol_util"].fillna(recomputed.round(1))
    out["bc_util_missing"] = out["bc_util"].isna().astype(int)
    out["bc_util"] = out["bc_util"].fillna(imp["bc_util_median"])
    out["ever_delinq"] = out["mths_since_last_delinq"].notna().astype(int)
    for c in ["revol_util", "bc_util"]:                                      # re-cap after filling
        out[f"{c}_cap"] = out[c].clip(upper=p["caps"][c]["upper"])

    out["fico_mid"] = (out["fico_range_low"] + out["fico_range_high"]) / 2   # A4 derived
    out["loan_to_income"] = (out["loan_amnt"] / out["annual_inc_cap"]).clip(upper=p["feature_caps"]["loan_to_income"])

    out["emp_length_years"] = (out["emp_length"].replace("< 1 year", "0 years")   # A5
                               .str.extract(r"(\d+)", expand=False).astype(float))
    out["purpose_grp"] = out["purpose"].where(out["purpose"].isin(p["purpose_keep"]), "other")
    out["verification_grp"] = out["verification_status"].map(VERIF_MAP)
    out["purpose_risk"] = out["purpose_grp"].map(PURPOSE_RISK)
    return out

# ---------- A6: PCA and factor analysis ----------
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PCA_COLS = [                                                # A6: 19 inputs, same order as notebook 06
    "acc_open_past_24mths", "annual_inc_log", "avg_cur_bal_log", "bc_util_cap", "credit_hist_months",
    "dti_cap", "fico_mid", "inq_last_6mths", "loan_amnt", "loan_to_income", "mort_acc",
    "num_tl_op_past_12m", "open_acc", "revol_bal_log", "revol_util_cap", "tot_cur_bal_log",
    "tot_hi_cred_lim_log", "total_acc", "total_rev_hi_lim_log",
]
PCA_LOG = ["inq_last_6mths", "mort_acc", "num_tl_op_past_12m", "open_acc",
           "credit_hist_months", "acc_open_past_24mths"]          # raw max |z| > 5 rule
FA_COLS = [c for c in PCA_COLS if c not in ("avg_cur_bal_log", "loan_to_income")]   # near-redundant / formula
N_PCA = 4
FACTOR_NAMES = ["debt_size", "card_utilization", "card_size", "credit_seeking"]     # promax F1..F4
PCA_SCORE_COLS = [f"pc{i}" for i in range(1, N_PCA + 1)]
FACTOR_SCORE_COLS = [f"factor_{n}" for n in FACTOR_NAMES]


def pca_matrix(df):
    """The 19 PCA/FA inputs, with log1p on the 6 count fields."""
    X = df[PCA_COLS].copy()
    X[PCA_LOG] = np.log1p(X[PCA_LOG])
    return X


def fit_latent(fit_df):
    """Learn the standardizer, PCA (4 components) and factor model (4 factors, promax) on <= cutoff rows."""
    X = pca_matrix(fit_df)
    scaler = StandardScaler().fit(X)
    Z = pd.DataFrame(scaler.transform(X), columns=PCA_COLS, index=X.index)
    pca = PCA(n_components=N_PCA, svd_solver="full").fit(Z)
    fa = FactorAnalyzer(n_factors=len(FACTOR_NAMES), rotation="promax", method="minres")
    fa.fit(Z[FA_COLS])
    return {"scaler": scaler, "pca": pca, "fa": fa}


def apply_latent(df, models):
    """Add PCA component scores and factor scores using the fitted models. Never re-learns anything."""
    Z = pd.DataFrame(models["scaler"].transform(pca_matrix(df)), columns=PCA_COLS, index=df.index)
    out = df.copy()
    out[PCA_SCORE_COLS] = models["pca"].transform(Z)
    out[FACTOR_SCORE_COLS] = models["fa"].transform(Z[FA_COLS])
    return out
# ---------- WoE bins for the scorecard (fitted on <= cutoff rows) ----------
TARGET = "defaulted"
N_WOE_BINS = 5
CONTRACT_VERSION = "1.0.0"


def woe_groups(s, rule):
    """Assign each value to its WoE bin label: '0'..'4' for numeric bins, the level for categories, 'missing'."""
    if rule["type"] == "numeric":
        edges = [-np.inf] + rule["edges"] + [np.inf]
        codes = pd.Series(pd.cut(s, edges, labels=False), index=s.index).astype("Int64")
        return codes.astype(str).replace("<NA>", "missing")
    groups = s.astype(str).where(s.notna(), "missing")
    if "mapping" in rule:
        groups = groups.map(rule["mapping"]).fillna(groups)
    return groups


def fit_woe(df, cols, target=TARGET, n_bins=N_WOE_BINS):
    """Learn bins and their WoE (ln(%good / %bad), +0.5 smoothing) for each column."""
    bins = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 12:
            _, edges = pd.qcut(s.dropna(), n_bins, retbins=True, duplicates="drop")
            rule = {"type": "numeric", "edges": [float(e) for e in edges[1:-1]]}     # inner edges only
        else:
            rule = {"type": "categorical"}
            if c == "purpose_grp":
                rule["mapping"] = PURPOSE_RISK
        t = pd.crosstab(woe_groups(s, rule), df[target])
        good, bad = t[0] + 0.5, t[1] + 0.5
        pg, pb = good / good.sum(), bad / bad.sum()
        woe = np.log(pg / pb)
        rule["woe"] = {str(k): float(v) for k, v in woe.items()}
        rule["iv"] = float(((pg - pb) * woe).sum())
        bins[c] = rule
    return bins


def apply_woe(df, bins):
    """Replace each column by its bin's WoE; unseen levels get 0 (neutral)."""
    out = pd.DataFrame(index=df.index)
    for c, rule in bins.items():
        out[f"woe_{c}"] = woe_groups(df[c], rule).map(rule["woe"]).fillna(0.0).astype(float)
    return out


# ---------- the whole contract transform in one object ----------
class CreditRiskTransform:
    """Raw application + bureau rows -> cleaned fields -> features -> PCA and factor scores (+ WoE).

    fit() learns every number on loans issued <= fit_cutoff only; transform() never re-learns anything.
    """
    contract_version = CONTRACT_VERSION

    def __init__(self, fit_cutoff=FIT_CUTOFF):
        self.fit_cutoff = pd.Timestamp(fit_cutoff)

    def fit(self, raw_df):
        cleaned = clean(raw_df)
        fit_rows = cleaned[cleaned["issue_dt"] <= self.fit_cutoff]
        self.params_ = fit_basic(fit_rows)
        basic = apply_basic(fit_rows, self.params_)
        self.models_ = fit_latent(basic)
        latent = apply_latent(basic, self.models_)
        self.woe_bins_ = fit_woe(latent, MODEL_FEATURES + FACTOR_SCORE_COLS)
        self.n_fit_rows_ = len(fit_rows)
        return self

    def transform(self, raw_df):
        return apply_latent(apply_basic(clean(raw_df), self.params_), self.models_)

    def woe(self, transformed_df):
        return apply_woe(transformed_df, self.woe_bins_)