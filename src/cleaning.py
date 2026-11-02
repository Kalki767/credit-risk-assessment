"""A1 cleaning rules as reusable code.

Two kinds of steps:
- dataset steps (training only): drop duplicate ids, merge applications with the bureau file
- row rules (every application, including the holdout and new applicants): clean()

The rules and their evidence are documented in notebooks/01_audit.ipynb and reports/audit_issues.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that must never reach the model: post-default information (A1 leakage finding).
LEAKAGE_COLS = ["bureau_recovery_flag"]

DATE_FORMATS = {
    r"[A-Za-z]{3}-\d{4}": "%b-%Y",     # Dec-2013
    r"\d{4}-\d{2}-\d{2}": "%Y-%m-%d",  # 2013-12-01
    r"\d{2}/\d{4}": "%m/%Y",           # 12/2013
}

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}
VALID_CODES = set(STATE_NAMES.values())

ZIP_FIRST_DIGIT = {
    "0": {"CT", "MA", "ME", "NH", "NJ", "RI", "VT"},
    "1": {"DE", "NY", "PA"},
    "2": {"DC", "MD", "NC", "SC", "VA", "WV"},
    "3": {"AL", "FL", "GA", "MS", "TN"},
    "4": {"IN", "KY", "MI", "OH"},
    "5": {"IA", "MN", "MT", "ND", "SD", "WI"},
    "6": {"IL", "KS", "MO", "NE"},
    "7": {"AR", "LA", "OK", "TX"},
    "8": {"AZ", "CO", "ID", "NM", "NV", "UT", "WY"},
    "9": {"AK", "CA", "HI", "OR", "WA"},
}


# ---------- helpers (identical to notebook 01) ----------

def parse_mixed_dates(s):
    """Parse each known date shape with its own format; anything else becomes NaT. Month precision."""
    s = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    for pattern, fmt in DATE_FORMATS.items():
        mask = s.str.fullmatch(pattern)
        out[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
    return out.dt.to_period("M").dt.to_timestamp()


def clean_state(s):
    """Map full names to codes and upper-case codes; anything unrecognized becomes NaN."""
    norm = s.astype(str).str.strip().str.lower()
    mapped = norm.map(STATE_NAMES).fillna(norm.str.upper())
    return mapped.where(mapped.isin(VALID_CODES))


def expected_installment(amount, rate_pct, months):
    """Standard loan payment formula: amount × r / (1 − (1 + r)^−n), r = monthly rate."""
    r = rate_pct / 100 / 12
    return amount * r / (1 - (1 + r) ** (-months))


def zip_state_mismatch(zip_s, state_s):
    """True where the ZIP's first digit doesn't belong to the state's region."""
    first = zip_s.astype(str).str[0]
    return pd.Series([st not in ZIP_FIRST_DIGIT.get(d, set())
                      for d, st in zip(first, state_s)], index=zip_s.index)


# ---------- dataset steps (training only) ----------

def load_training_raw(raw_dir):
    """Applications (duplicate ids dropped, keep first) merged with the bureau file."""
    raw_dir = Path(raw_dir)
    apps = pd.read_csv(raw_dir / "applications.csv", dtype={"id": str}, low_memory=False)
    bureau = pd.read_csv(raw_dir / "credit_bureau.csv", dtype={"id": str}, low_memory=False)
    apps = apps.drop_duplicates(subset="id", keep="first").reset_index(drop=True)
    return apps.merge(bureau, on="id", how="inner")


# ---------- row rules (every application) ----------

def clean(df):
    """Apply the A1 row rules to merged application + bureau rows. Returns a new DataFrame."""
    out = df.copy()

    # formats
    out["issue_dt"] = parse_mixed_dates(out["issue_d"])
    out["earliest_cr_dt"] = parse_mixed_dates(out["earliest_cr_line"])
    out["addr_state_clean"] = clean_state(out["addr_state"])
    out["term_months"] = out["term"].str.extract(r"(\d+)", expand=False).astype(int)

    # loan amount recorded 10x too large (installment is exactly 1/10 of what the amount requires)
    ratio = out["installment"] / expected_installment(out["loan_amnt"], out["int_rate"], out["term_months"])
    x10 = ratio.between(0.099, 0.101)
    out["loan_amnt_corrected"] = x10.astype(int)
    out.loc[x10, "loan_amnt"] = out.loc[x10, "loan_amnt"] / 10

    # income entered in thousands
    low = out["annual_inc"] < 1000
    out["annual_inc_corrected"] = low.astype(int)
    out.loc[low, "annual_inc"] = out.loc[low, "annual_inc"] * 1000

    # sign-flipped utilization
    for col in ["revol_util", "bc_util"]:
        neg = out[col] < 0
        out[f"{col}_corrected"] = neg.astype(int)
        out.loc[neg, col] = out.loc[neg, col].abs()

    # swapped FICO bounds
    bad = out["fico_range_high"] < out["fico_range_low"]
    lo, hi = out["fico_range_low"], out["fico_range_high"]
    out["fico_swapped"] = bad.astype(int)
    out["fico_range_low"], out["fico_range_high"] = np.minimum(lo, hi), np.maximum(lo, hi)

    # first credit line after the loan was issued: impossible -> missing (imputed later)
    after = out["earliest_cr_dt"] > out["issue_dt"]
    out["earliest_cr_invalid"] = after.astype(int)
    out.loc[after, "earliest_cr_dt"] = pd.NaT

    # remaining installment mismatches (after the x10 fix): flag only
    out["inst_ratio"] = out["installment"] / expected_installment(
        out["loan_amnt"], out["int_rate"], out["term_months"])
    out["installment_mismatch"] = (~out["inst_ratio"].between(0.98, 1.02)).astype(int)

    # ZIP region inconsistent with state: flag only
    out["zip_state_mismatch"] = zip_state_mismatch(out["zip_code"], out["addr_state_clean"]).astype(int)

    # leakage: never let post-default information through
    return out.drop(columns=LEAKAGE_COLS, errors="ignore")