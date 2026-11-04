# Model card: credit-risk PD scorecard v1.0.0

## Purpose
Predicts the probability of default (PD) of an unsecured personal loan at application time, to auto-approve, send to manual review or decline. Built on the Part A data contract (contract_version 1.0.0).

## Data window
- Training: loans issued 2013-01 to 2014-12 (14,373). Validation, used once: 2015 (15,627). Holdout: 2016 (5,000, unlabelled), scored in `reports/holdout_scores.csv`.
- All transforms (cleaning, caps, WoE bins, PCA, factors) were fitted only on loans issued up to 2014-12-31 and used unchanged.
- After the evaluation, a 2-parameter recalibration was learned on 2015 outcomes (production refresh, see below).

## Model
- **Production model:** WoE scorecard: logistic regression (C = 1) on 14 contract WoE features: term, FICO, loan-to-income, accounts opened in 24 months, DTI, accounts opened in 12 months, verification, average balance, card utilization, income, inquiries (6 months), home ownership, total card limits, purpose.
- **Challenger:** XGBoost (depth 1, 200 trees, learning rate 0.1, monotonic constraints, Platt calibration).
- **Choice rule (set in DESIGN.md before training):** XGBoost only if it beats the scorecard by at least 0.01 AUC on 2015. It beat it by +0.0044, so the scorecard was chosen; it is also fully explainable.
- **Not used:** grade, sub_grade, int_rate, installment (the old model's outputs); zip code and state (fairness, no signal); job title (definition changed in 2013Q4); everything from loan servicing and bureau_recovery_flag (post-issue leakage).

## Metrics
| | CV AUC (2014 quarters) | 2015 AUC | 2015 Gini | 2015 KS |
| --- | --- | --- | --- | --- |
| Scorecard | 0.6975 | **0.7125** | 0.425 | 0.309 |
| XGBoost | 0.7065 | 0.7169 | 0.434 | 0.318 |
| Grade alone | 0.688 | 0.7026 | 0.405 | 0.299 |

- By 2015 quarter: AUC 0.693 / 0.719 / 0.716 / 0.718, above grade in every quarter; PSI of PD vs 2014 at most 0.011.
- Calibration: before recalibration, mean PD 17.2% vs 20.1% actual on 2015 (36-month loans gap 1.4 points, 60-month 7.6).

## Decision policy
- Economics: a repaid loan earns 10% of the amount, a default loses 70% → break-even PD 12.5%.
- **Evaluation (honest test):** cut-off chosen on 2014 out-of-fold PDs (13.5%); bands with an assumed 10% review capacity: approve < 12.4%, review 12.4–14.7%, decline ≥ 14.7%. On 2015: approve 40.3% (9.7% default), +$1.51M, vs −$17.0M for approving everyone.
- **Production:** PD = expit(1.079 × logit(scorecard PD) + 0.327), learned on 2015 outcomes after the evaluation (monotone: ranking and reason codes unchanged; conservative, because 2015 labels over-count defaults). Bands centred on break-even: **auto-approve PD < 11.3%, review 11.3–13.8%, auto-decline ≥ 13.8%**. On 2015 (in-sample, a check only): approve 27.8%, 7.7% default.
- **Review triggers:** joint application, FICO below the training range, DTI above the training range send an auto-approval to review.

## Ablation and which data-science outputs helped
- XGBoost CV AUC: raw features 0.7065 · factor scores only 0.6342 · both 0.7084 (+0.0019, below the 0.002 rule) → raw features only.
- **Helped prediction:** cleaning and unit fixes, engineered features (loan-to-income, FICO midpoint, credit history), caps and imputation, WoE bins (the scorecard is built on them), leakage removal, category groupings.
- **Helped interpretation, not prediction:** the four factor scores (debt size, card utilization, card size, credit seeking); they compress 17 columns into 4 traits holding about two-thirds of the ranking signal.
- Cost of independence from the old model: adding grade + int_rate would add +0.009 CV AUC.

## Error analysis and drift
- **Weak segments (2015):** 60-month loans under-predicted by 7.6 points (mainly label bias); income ≥ 120k under-predicted (36-month gap 4.6 points), the same for verified and unverified income (inflated income ruled out), probably because income enters twice (income and loan-to-income); purpose "other" 5.6 points. No segment over-predicted. The safest 10% are well calibrated (4.5% vs 4.7%).
- **Fairness:** 4 US regions: approval 37–42%, default among approved 8.6–10.2% with overlapping CIs. Income bands: lower approval for lower incomes, with calibration gaps similar to the average.
- **Grade comparison:** grade's best policy (A–B) earned more on 2015 (+$1.66M vs +$1.39M at a single cut-off): grade is better at picking the very safest loans, but inside grade A the scorecard still separates 2.8% from 7.0% default.
- **Holdout 2016 vs 2015:** PSI of PD 0.011 and all top-10 inputs ≤ 0.024 (stable); decision mix approve 28.5% / review 10.4% / decline 61.1% vs 27.8 / 10.0 / 62.2.
- **Alerts:** PSI ≥ 0.10 investigate, ≥ 0.25 act; decision mix shifts of more than 5 points; and predicted vs realized default rate by monthly vintage as loans mature (PSI cannot detect concept drift: 2015 applicants looked the same as 2014's but defaulted more).

## Inference
`python -m src.predict applicant.json` or `predict(dict)` → pd, decision, decision_basis, model_version, reason_codes (top 4 features by risk above their best bin, in plain words), review triggers, metadata. Invalid input is rejected with a clear message. Latency (600 end-to-end calls on a 2017 laptop, Dell Latitude 5480): **p50 149 ms, p95 221 ms, above the 100 ms target.** Profiling shows no single bottleneck: the time is spread over hundreds of small pandas operations in the per-row transform (cleaning, caps, PCA and factor scores, WoE), each with a fixed cost regardless of row count; the scorecard itself is 14 multiplications, and artifacts load once at start-up (about 1.4 s). Fix: skip the unused PCA/factor step when serving, and build a lean single-row feature path from data_spec.json, proven identical by the parity test.

## Known limits
- Trained on accepted loans only: outcomes of declined applicants are unknown (reject inference not done).
- In-progress bias: loans still running were removed, so recent and 60-month labels over-count defaults; 60-month PDs are the least reliable.
- High incomes are over-credited; the production recalibration was fitted and checked on the same year (2015).
- US Lending Club loans from 2013–2015; application data only, so the ranking ceiling is moderate (AUC about 0.71).

## Next steps
Recalibrate on matured 2015 loans; limit the credit given to income above about 120k; test overall utilization, card debt ÷ income and income × term features; test combining the score with grade; reject inference; monitor by vintage.