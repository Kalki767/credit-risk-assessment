"""Ranking and probability metrics used for every model."""
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ranking_metrics(y, score):
    """AUC, Gini and KS for a score where higher = riskier."""
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)          # tpr = share of defaulters passed, fpr = share of payers passed
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}


def probability_metrics(y, pd_hat):
    """Brier score (mean squared error of the probabilities) and mean predicted vs actual default rate."""
    return {"brier": brier_score_loss(y, pd_hat), "mean_pd": float(np.mean(pd_hat)), "default_rate": float(np.mean(y))}

def psi(expected, actual, n_bins=10, eps=1e-4):
    """Population Stability Index of `actual` vs `expected`.
    Numeric with many values: decile bins of expected (+ a missing bin); otherwise one bin per category."""
    e, a = pd.Series(expected), pd.Series(actual)
    if pd.api.types.is_numeric_dtype(e) and e.nunique() > n_bins:
        inner = np.unique(np.quantile(e.dropna(), np.linspace(0, 1, n_bins + 1)[1:-1]))
        to_bin = lambda s: pd.Series(np.where(s.isna(), -1, np.searchsorted(inner, s.fillna(0), side="right")))
    else:
        to_bin = lambda s: s.astype(str).where(s.notna(), "missing")
    pe, pa = to_bin(e).value_counts(normalize=True), to_bin(a).value_counts(normalize=True)
    bins = pe.index.union(pa.index)
    pe, pa = pe.reindex(bins, fill_value=0).clip(lower=eps), pa.reindex(bins, fill_value=0).clip(lower=eps)
    return float(((pa - pe) * np.log(pa / pe)).sum())