"""The production PD model: WoE scorecard + recalibration layer + decision bands."""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression

from .features import woe_groups

DECISIONS = ("auto-approve", "review", "auto-decline")


def fold_woe(fit_df, apply_df, features, rules, target):
    """Contract bin edges; WoE values learned from fit_df only (+0.5 smoothing); unseen bins -> 0."""
    out = pd.DataFrame(index=apply_df.index)
    for c in features:
        t = pd.crosstab(woe_groups(fit_df[c], rules[c]), fit_df[target])
        good, bad = t[0] + 0.5, t[1] + 0.5
        woe = np.log((good / good.sum()) / (bad / bad.sum()))
        out[f"woe_{c}"] = woe_groups(apply_df[c], rules[c]).map(woe).fillna(0.0).astype(float)
    return out


class ScorecardModel:
    """Logistic regression on WoE values, then PD = expit(slope * logit(raw PD) + intercept), then bands."""

    def __init__(self, features, C=1.0, version="1.0.0"):
        self.features, self.C, self.version = list(features), C, version
        self.cal_slope, self.cal_intercept = 1.0, 0.0      # identity until recalibrated
        self.t1 = self.t2 = None

    @property
    def woe_cols(self):
        return [f"woe_{c}" for c in self.features]

    def fit(self, woe_df, y):
        self.lr_ = LogisticRegression(C=self.C, max_iter=1000).fit(woe_df[self.woe_cols], y)
        self.coef_ = dict(zip(self.features, self.lr_.coef_[0]))
        return self

    def raw_pd(self, woe_df):
        return self.lr_.predict_proba(woe_df[self.woe_cols])[:, 1]

    def fit_recalibration(self, raw_pd, y):
        cal = LogisticRegression(C=1e6, max_iter=1000).fit(logit(np.asarray(raw_pd)).reshape(-1, 1), y)
        self.cal_slope, self.cal_intercept = float(cal.coef_[0, 0]), float(cal.intercept_[0])
        return self

    def recalibrate(self, raw_pd):
        return expit(self.cal_slope * logit(np.asarray(raw_pd)) + self.cal_intercept)

    def predict_pd(self, woe_df):
        return self.recalibrate(self.raw_pd(woe_df))

    def set_thresholds(self, t1, t2):
        self.t1, self.t2 = float(t1), float(t2)

    def decide(self, pd_):
        p = np.asarray(pd_)
        return np.where(p < self.t1, DECISIONS[0], np.where(p < self.t2, DECISIONS[1], DECISIONS[2]))