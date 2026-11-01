# Audit notes

## Resolved in A1 (details in reports/audit_issues.csv)
- [x] Duplicate ids (600) → dropped
- [x] issue_d / earliest_cr_line mixed formats (1,200 each) → parsed to month
- [x] addr_state names / case (1,963) → two-letter codes
- [x] loan_amnt ×10 (300) → ÷10, flagged; confirmed by servicing principal
- [x] annual_inc in thousands (899) → ×1,000, flagged
- [x] revol_util (295) / bc_util (290) sign flipped → absolute value, flagged
- [x] FICO bounds swapped (150) → fixed, flagged
- [x] earliest_cr_line after issue (300) → missing, flagged
- [x] Installment mismatches (11) → kept, flagged
- [x] dti > 40 → genuine, kept
- [x] bureau_recovery_flag → leakage, excluded
- [x] Label vs servicing → consistent (1 plausible case kept)
- [x] title → excluded (duplicates purpose)
- [x] ZIP/state mismatch (27) → kept, flagged
- [x] Joint apps 16 → 96 in holdout → flagged, drift
- [x] 'wedding' absent from holdout → merge in A5

## Pending
- [ ] A2: default rate by month and term; in-progress-loan bias
- [ ] A3: outliers and heavy tails (income 7.5M, revol_util 366.6%, balances); caps fitted ≤2014
- [ ] A4: reliability ratings; missingness vs default (mths_since_last_delinq 45%, emp_title/emp_length ~6%,
      bc_util 1%); impute earliest_cr_line (300); leakage list for the model; new features
- [ ] A5: group emp_title into job families; order emp_length; merge rare levels (incl. wedding); WoE/IV
- [ ] Code all A1 rules into src/cleaning.py and check they reproduce these counts

## Checks passed
- No duplicates in bureau, servicing or holdout; ids match across files; no holdout overlap
- Categories (other than addr_state) have no case or spacing variants
- Servicing and holdout dates use one format; ZIP format valid everywhere
- int_rate on one scale; FICO within 300–850; count relationships consistent
- Holdout free of all planted errors; no unseen categories

## Observations
- Default rate 18.7% (5,617 / 30,000); FICO minimum 660 (accepted-only bias)
- Loans per year: 2013 = 5,454, 2014 = 9,212, 2015 = 15,934, holdout 2016 = 5,000
- 2016 drift: joint applications ×6; individual DTI above 40

## A3: Financial values and outliers (fit sample: loans issued ≤ 2014, n = 14,373)

### Distributions
- Money fields (annual_inc, revol_bal, tot_cur_bal, avg_cur_bal, tot_hi_cred_lim, total_rev_hi_lim) are right-skewed across orders of magnitude (annual_inc skew 63.3). log1p makes them near-symmetric.
- loan_amnt, installment, dti, revol_util and bc_util are bounded or near-symmetric; log1p over-corrects them into left skew (e.g. dti 0.19 → −1.1). They are kept linear.
- loan_amnt spikes at round amounts and piles at $35,000 (the platform maximum). dti stops at 40 (the policy limit for individual applicants).
- tot_cur_bal, avg_cur_bal and tot_hi_cred_lim are bimodal on the log scale; revolving-only total_rev_hi_lim is not. Medians by home_ownership confirm mortgage debt as the cause: MORTGAGE vs RENT 4.5–6.1× on those fields vs 1.4× on total_rev_hi_lim, and OWN is close to RENT.
- Zero groups (revol_bal 40, revol_util 49, bc_util 115) sit apart from the main distribution; handled with flags in A4.

### Outlier method (chosen per field type)
- Compared mean ± 3 std, the IQR fence, p99 and p99.5.
- Mean ± 3 std fails on skewed data: removing one income (of 14,373) cuts the std from 76,973 to 45,695 and moves the limit from $305k to $211k (masking); the lower limit is a negative income.
- The IQR fence flags 428–829 normal borrowers on money fields.
- Percentile caps cut real values on bounded fields (p99 dti = 34.9 would cap 143 loans inside the policy limit).
- **Decision:** money fields get log1p + cap at p99.5; dti, revol_util and bc_util are capped at the IQR fence (40.95 / 125 / 150); loan_amnt and installment are not capped. Caps are fitted on the ≤ 2014 sample and stored in reports/a3_caps.json (→ data_spec.json).

### Extremes checked
- revol_util 366.6% is genuine as recorded: $2,566 / $700 limit. The recomputation agrees within 5 pts for 96.2% of loans. Capped (2 loans).
- annual_inc $7.5M (sales rep, Verified) and $1.25M are suspected extra zeros (they imply DTI 20 / 18.2 at 100× / 10× lower income). Not provable, so capped and not corrected. "Verified" does not guarantee accuracy (input to A4).

### Effect of the treatment
- Under 0.5% of values changed; annual_inc skew 63.33 → 0.09.
- Ranking preserved (raw and log deciles identical).
- Capped loans stay separable at the cap. Big-balance borrowers default less (4–7% vs 11–14%).
- Drift: 2015–16 has 106 / 111 loans above the 2014 caps for revol_bal / total_rev_hi_lim vs ~78 expected. Also 4 later dti > 40.95 (joint applications) and 2 bc_util > 150. Check with PSI in A7.