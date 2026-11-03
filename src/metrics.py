"""Ranking and probability metrics used for every model."""
import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def ranking_metrics(y, score):
    """AUC, Gini and KS for a score where higher = riskier."""
    auc = roc_auc_score(y, score)
    fpr, tpr, _ = roc_curve(y, score)          # tpr = share of defaulters passed, fpr = share of payers passed
    return {"auc": auc, "gini": 2 * auc - 1, "ks": float(np.max(tpr - fpr))}


def probability_metrics(y, pd_hat):
    """Brier score (mean squared error of the probabilities) and mean predicted vs actual default rate."""
    return {"brier": brier_score_loss(y, pd_hat), "mean_pd": float(np.mean(pd_hat)), "default_rate": float(np.mean(y))}