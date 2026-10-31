- id: anonymized 10-char hex (text is correct). applications has 30,356 rows vs 29,758 in bureau/servicing:
  check for duplicate ids and ids missing from other files
  Dates stored as text: issue_d, earliest_cr_line, last_pymnt_d, last_credit_pull_d. Parse them and check for mixed formats
- term → 36/60; emp_length → 0–10 ordered
- int_rate, revol_util are numeric: check for mixed scales (0.13 vs 13)
- bureau_recovery_flag: extra column, possible leakage (held for later)

## Missing values (first look)
| Column | File(s) | % missing | Likely meaning | Check / plan |
|---|---|---|---|---|
| mths_since_last_delinq | bureau 45.1, holdout 47.6 | never delinquent (structural) | default rate missing vs present; fill + flag, not median |
| emp_title | apps 6.0, holdout 6.8 | not provided | default rate missing vs present; "unknown" group |
| emp_length | apps 5.5, holdout 6.7 | not provided | overlap with emp_title; default rate |
| bc_util | bureau 1.1, holdout 1.3 | no bankcards | fill + "no bankcard" flag (≠ 0%) |
| last_pymnt_d | servicing 0.1 | never paid | label check only, not a feature |
| title | holdout only 5.5 | not missing in training | transform must handle unseen missing |
| revol_util | holdout only 0.1 | not missing in training | transform must handle unseen missing |

- Also look for disguised missing values ("n/a", "", "unknown", 0) in categories and numbers.

## Numeric first look
- Row counts: applications 30,600 vs bureau 30,000 → 600 extra rows (duplicates? missing bureau records?)
- Default rate 18.7%. FICO min = 660 (accepted-only bias visible)
- Errors: loan_amnt max 350,000 (brief max 40,000); annual_inc min 11.5 / p1 = 50 (thousands?);
  revol_util min -101.3; bc_util min -102.4
- Investigate: revol_util max 366.6; dti max 100.09; annual_inc max 7.5M; fico_high ≤ fico_low?;
  total_rev_hi_lim = 0 with balance > 0?; bankruptcies > pub_rec?; num_tl_op_12m > acc_open_24m?
- Heavy tails (cap/log in A3): revol_bal, tot_cur_bal, tot_hi_cred_lim, avg_cur_bal, total_rev_hi_lim
- int_rate scale consistent (5.32–28.99)

- term, grade: clean. Default rate 18.7% (5,731 / 30,600)
- Loans per year: 2013 = 5,454; 2014 = 9,212 (train 14,666); 2015 = 15,934 (validation)
- issue_d parsed fully with format="mixed", but check that the raw formats are consistent

- emp_length: 11 clean levels + 1,693 NaN (5.5%)
- home_ownership: MORTGAGE / RENT / OWN only (no rare levels)
- application_type: 16 Joint App rows. Flag or merge; income may be joint
- Main categories clean → planted issues likely in emp_title, title, zip/state, holdout, cross-file