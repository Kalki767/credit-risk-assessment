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