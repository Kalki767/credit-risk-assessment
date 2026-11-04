"""Write three made-up applicants (low, borderline, high risk) to examples/ and score them.
No real rows are used. The borderline profile is tuned (FICO x DTI grid) to land in the review band."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import predict   # noqa: E402


def installment(amount, rate_pct, months):
    """Monthly payment of a standard amortizing loan."""
    r = rate_pct / 100 / 12
    return round(amount * r / (1 - (1 + r) ** -months), 2)


def applicant(**fields):
    """Fill in the installment so it matches amount, rate and term (avoids the A1 unit-error rule)."""
    months = int(fields["term"].split()[0])
    fields["installment"] = installment(fields["loan_amnt"], fields["int_rate"], months)
    return fields


LOW = applicant(
    id=900001, issue_d="Nov-2026", loan_amnt=10000, term=" 36 months", int_rate=7.5, grade="A", sub_grade="A2",
    purpose="credit_card", title="Credit card refinancing", application_type="Individual", emp_title="Teacher",
    emp_length="10+ years", home_ownership="MORTGAGE", annual_inc=120000, verification_status="Not Verified",
    dti=8.0, zip_code="945xx", addr_state="CA", earliest_cr_line="Mar-2008", fico_range_low=780,
    fico_range_high=784, inq_last_6mths=0, delinq_2yrs=0, mths_since_last_delinq=None, pub_rec=0,
    pub_rec_bankruptcies=0, open_acc=10, total_acc=30, revol_bal=7200, revol_util=12.0, bc_util=15.0,
    tot_cur_bal=250000, avg_cur_bal=25000, total_rev_hi_lim=60000, tot_hi_cred_lim=330000, mort_acc=2,
    acc_open_past_24mths=2, num_tl_op_past_12m=0)

BORDERLINE = applicant(
    id=900002, issue_d="Nov-2026", loan_amnt=12000, term=" 36 months", int_rate=11.5, grade="B", sub_grade="B5",
    purpose="debt_consolidation", title="Debt consolidation", application_type="Individual", emp_title="Nurse",
    emp_length="4 years", home_ownership="RENT", annual_inc=70000, verification_status="Source Verified",
    dti=15.0, zip_code="100xx", addr_state="NY", earliest_cr_line="Jun-2014", fico_range_low=700,
    fico_range_high=704, inq_last_6mths=1, delinq_2yrs=0, mths_since_last_delinq=None, pub_rec=0,
    pub_rec_bankruptcies=0, open_acc=9, total_acc=20, revol_bal=14000, revol_util=56.0, bc_util=62.0,
    tot_cur_bal=80000, avg_cur_bal=8900, total_rev_hi_lim=25000, tot_hi_cred_lim=110000, mort_acc=0,
    acc_open_past_24mths=4, num_tl_op_past_12m=2)

HIGH = applicant(
    id=900003, issue_d="Nov-2026", loan_amnt=30000, term=" 60 months", int_rate=22.0, grade="E", sub_grade="E2",
    purpose="small_business", title="Business", application_type="Individual", emp_title="Driver",
    emp_length="< 1 year", home_ownership="RENT", annual_inc=40000, verification_status="Verified",
    dti=32.0, zip_code="770xx", addr_state="TX", earliest_cr_line="Jan-2022", fico_range_low=662,
    fico_range_high=666, inq_last_6mths=4, delinq_2yrs=1, mths_since_last_delinq=12, pub_rec=1,
    pub_rec_bankruptcies=1, open_acc=12, total_acc=15, revol_bal=9000, revol_util=90.0, bc_util=95.0,
    tot_cur_bal=20000, avg_cur_bal=1670, total_rev_hi_lim=10000, tot_hi_cred_lim=25000, mort_acc=0,
    acc_open_past_24mths=10, num_tl_op_past_12m=5)


def tune_borderline(base):
    """Try FICO x DTI combinations; keep the one whose PD is closest to the middle of the review band."""
    best = None
    for fico in (665, 677, 690, 707, 730, 760):
        for dti in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0):
            app = {**base, "fico_range_low": fico, "fico_range_high": fico + 4, "dti": dti}
            out = predict(app)
            t = out["metadata"]["thresholds"]
            centre = (t["auto_approve_below"] + t["auto_decline_from"]) / 2
            if out["decision"] == "review" and (best is None or abs(out["pd"] - centre) < best[0]):
                best = (abs(out["pd"] - centre), app)
    if best is None:
        raise SystemExit("no FICO x DTI combination landed in the review band - send me this message")
    return best[1]


if __name__ == "__main__":
    out_dir = ROOT / "examples"
    out_dir.mkdir(exist_ok=True)
    for name, app in [("low_risk", LOW), ("borderline", tune_borderline(BORDERLINE)), ("high_risk", HIGH)]:
        with open(out_dir / f"{name}.json", "w") as f:
            json.dump(app, f, indent=2)
        res = predict(app)
        print(f"{name:10s} FICO {app['fico_range_low']} DTI {app['dti']}: pd {res['pd']:.4f} -> {res['decision']}"
              f" | triggers {res['review_triggers']} | reasons {res['reason_codes']}")