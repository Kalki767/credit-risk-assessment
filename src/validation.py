"""Fail-fast checks: forbidden (post-issue) columns and input validation against data_spec.json."""
import pandas as pd


def check_no_forbidden(df, features, forbidden):
    """Stop if any post-issue column is in the data or among the model features."""
    in_data = sorted(set(forbidden) & set(df.columns))
    in_features = sorted(set(forbidden) & set(features))
    if in_data or in_features:
        raise ValueError(f"forbidden columns present - data: {in_data}, features: {in_features}")


def validate_frame(df, spec, roles=("model input",)):
    """List every violation of data_spec.json (cleaned values): missing required, type, range, category."""
    problems = []
    for f in spec["fields"]:
        name = f["name"]
        if f.get("role") not in roles or name not in df.columns:
            continue
        s = df[name]
        if f.get("missing_strategy") == "required" and s.isna().any():
            problems.append(f"{name}: {int(s.isna().sum())} missing values (required)")
        v = s.dropna()
        if f["type"] in ("float", "int"):
            if not pd.api.types.is_numeric_dtype(v):
                problems.append(f"{name}: expected a number")
                continue
            if f.get("min") is not None and (v < f["min"]).any():
                problems.append(f"{name}: {int((v < f['min']).sum())} values below min {f['min']}")
            if f.get("max") is not None and (v > f["max"]).any():
                problems.append(f"{name}: {int((v > f['max']).sum())} values above max {f['max']}")
        elif f["type"] == "category" and f.get("allowed_values"):
            bad = sorted(set(v.astype(str)) - set(map(str, f["allowed_values"])))
            if bad:
                problems.append(f"{name}: values not allowed {bad[:5]} (allowed: {f['allowed_values']})")
    return problems


def validate_or_fail(df, spec):
    """Raise one error listing all problems, so bad data never reaches the model."""
    problems = validate_frame(df, spec)
    if problems:
        raise ValueError("input validation failed:\n- " + "\n- ".join(problems))