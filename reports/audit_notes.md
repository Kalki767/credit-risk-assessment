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

## A4: Reliability, missingness, leakage and features (fit sample: loans issued ≤ 2014)

### Missingness
- No missing group differs from noise in default rate, so treatments follow the reason for the gap.
- mths_since_last_delinq (46%): structural (never delinquent), handled with the `ever_delinq` flag and NaN kept. Never-delinquent borrowers default only 1 pt less.
- emp_title / emp_length: 685 of 686 missing emp_length also lack emp_title. Becomes an "Unknown" category (A5).
- bc_util (1.1%): likely no bankcard (unverifiable; rate stable across years at 1.05% vs 1.09%). Flag + median fill (69.4).
- revol_util (4): recomputed as balance / limit; 0 where balance = limit = 0.
- earliest_cr_dt (300 impossible dates): borrowers match normal ones (total_acc 24 vs 24; 54% have a past delinquency), so only the date is wrong.
  - Imputed as credit_hist_months using the median by total_acc quintile (139–210 months).
  - 5 groups chosen by validation on 2015: MAE 68.7 (global median) → 65.7 (5 groups) → 65.2 (20 groups).
  - A floor of months since the last delinquency applies to imputed rows only. No contradictions exist in observed rows.

### Reliability
- Round-number share: annual_inc 84% vs revol_bal / tot_cur_bal 0.1% (the chance rate) → self-reported vs exact bureau records.
- Verification does not change rounding (85% / 86% / 81%).
- Verified loans default more (19.3% vs 12.4%) and are larger ($16k vs $10k median): selective verification makes verification_status confounded, not a measure of accuracy.
- Ratings:
  - High: platform fields (installment matches the formula for 99.96% of loans) and bureau fields
  - Medium: home_ownership, dti, verification_status
  - Low: annual_inc, emp_title, emp_length

### Leakage
- Single-feature AUC: bureau_recovery_flag 0.906 vs ≤ 0.681 for every other column. Confirmed leakage, excluded.
- grade, sub_grade and int_rate are excluded as old-model output (benchmark and ablation in Part B).
- Audit flags are excluded (planted errors are absent from the holdout); so are zip_code, identifiers, raw dates, and raw/intermediate versions of treated fields.
- Every column is classified with a reason: 72 = 40 excluded + 2 pending (emp_title → A5, addr_state_clean → A7) + 33 features (after fico_mid replaces the FICO pair and the 3 derived features are added; saved in reports/a4_spec.json).

### Derived features
- loan_to_income (AUC 0.591) beats both its parts (0.538 / 0.552) and FICO.
- installment_to_income (0.575) captures the new loan's burden, which dti excludes.
- fico_mid replaces the FICO pair.
- revol_bal_zero dropped (AUC 0.501).

### Leakage
- Single-feature AUC: bureau_recovery_flag 0.906 vs ≤ 0.681 for every other column. Confirmed leakage, excluded.
- Old-model pricing excluded: grade, sub_grade, int_rate, and installment (Spearman ≈ 0.999 with int_rate within term). Benchmark and ablation in Part B.
- Audit flags (planted errors, absent from the holdout), zip_code, identifiers, raw dates and superseded raw/capped versions are excluded.
- All 76 columns classified: 31 features, 43 excluded with reasons, 2 pending (emp_title → A5, addr_state_clean → A7). Saved in reports/a4_spec.json.

### Derived features
- Kept: fico_mid (replaces the FICO pair) and loan_to_income (AUC 0.591, beats loan_amnt 0.538, income 0.552 and FICO 0.586).
- Rejected with evidence: installment_to_income (carries int_rate), principal_per_year_to_income (AUC 0.532; long-term risk is selection, not yearly burden), revol_bal_zero (redundant, 0.3%).

## A5: Categorical fields, WoE and IV (fit sample: loans issued ≤ 2014)

### Level review
- Every categorical field profiled (loans, share, default rate per level).
- application_type is constant in the fit sample (Individual only) → excluded. Joint applications (2015+, 96 in the holdout) are unseen by the model → monitoring / manual review.

### Transformations
- **purpose** → purpose_grp: levels with < 100 loans merged into "other" (moving, vacation, house, wedding, renewable_energy); unseen purposes also → "other". 8 groups.
- **emp_length** → emp_length_years (0–10, NaN = unknown).
- **emp_title** → job_family:
  - 10 families by ordered keyword rules (first match wins), stored in the spec
  - 15 test titles pass
  - 65.5% matched; the rest are one-off titles or employer names
- **verification_status** → verification_grp: Verified + Source Verified merged (WoE −0.138 vs −0.118; IV 0.0458 → 0.0457).
- **home_ownership** kept as 3 levels (RENT −0.170, OWN 0.011, MORTGAGE 0.139; all ≥ 1,406 loans; no new levels in 2015).

### WoE / IV
- 10 equal-count bins for numeric fields, missing as its own bin, +0.5 smoothing.
- sub_grade 0.503 / grade 0.454 / int_rate 0.451: the old model (confirms exclusion, benchmark).
- Honest features at most medium: term 0.195, fico_mid 0.108, loan_to_income 0.101.
- Delinquency and public-record fields are useless (≤ 0.004), likely due to accepted-only selection (A7).

### Purpose risk levels (scorecard coarse classing)
- high / medium / low.
- Within-level chi-square p = 0.855 / 0.942; between p = 4e-10.
- Out-of-time AUC (WoE learned 2013–14, applied to 2015): 0.5415 vs 0.5443 for 8 groups, within the 0.005 rule set in advance.
- Monitor medical (20.0% → 29.5%).

### Employment fields, out-of-time test (rule: AUC < 0.53 → exclude)
- emp_length_years 0.5076 → excluded.
- job_family 0.5438 despite IV 0.014 ("useless" by rule of thumb) → kept under fairness review: stays only if it adds ≥ 0.005 validation AUC in the Part B ablation; excluded from the scorecard.

### Rules
- An unseen category level gets WoE 0 (neutral).
- Features: 30. Pending: addr_state_clean (A7). Saved in reports/a5_spec.json.