"""The money rules: realized profit of a policy, the most profitable cut-off, and the review band."""
import numpy as np


def policy_result(approve, y, amount, gain, loss):
    """Profit of approving the loans where approve is True (+gain x amount if repaid, -loss x amount if defaulted)."""
    approve, y, amount = np.asarray(approve, bool), np.asarray(y), np.asarray(amount, float)
    pnl = np.where(y == 1, -loss, gain) * amount
    vol = amount[approve].sum()
    return {"approval_rate": float(approve.mean()),
            "default_rate_approved": float(y[approve].mean()) if approve.any() else float("nan"),
            "profit_M": float(pnl[approve].sum() / 1e6),
            "return_on_approved": float(pnl[approve].sum() / vol) if vol else float("nan")}


def best_cutoff(pd_, y, amount, gain, loss, grid):
    """The cut-off (approve if PD below it) with the highest total profit."""
    cuts = np.round(np.arange(*grid), 3)
    profits = [policy_result(np.asarray(pd_) < c, y, amount, gain, loss)["profit_M"] for c in cuts]
    return float(cuts[int(np.argmax(profits))])


def band_thresholds(pd_, centre, review_share):
    """T1, T2 so that review_share of applications fall around the centre (half on each side)."""
    p = np.asarray(pd_)
    q = (p < centre).mean()
    return float(np.quantile(p, q - review_share / 2)), float(np.quantile(p, q + review_share / 2))