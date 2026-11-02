# DESIGN: Probability-of-default model

Written before any model code. Updated at the end under "What changed".

## Problem framing

Predict each applicant's **probability of default (PD)** at application time, using only the contract's 29 application-time features (`contract/data_spec.json`). The PD drives a decision: **auto-approve, manual review or auto-decline**. A repaid loan earns 10% of `loan_amnt` and a defaulted loan loses 70%, so expected profit per loan is `loan_amnt × (0.10 − 0.80 × PD)` and the break-even PD is **12.5%**. The model must therefore rank applicants well *and* give calibrated probabilities, because the PD is compared with a money threshold.

## Data and time split

`contract/transform.joblib` is the first step of training and inference and is **never refitted**. Nothing from `loan_servicing.csv` reaches the model.

| Set | Loans | Use |
| --- | --- | --- |
| Train | issued ≤ 2014-12-31 (14,373) | fit models; time-aware cross-validation for tuning and calibration |
| Validation | 2015 (15,627) | final evaluation, used once |
| Holdout | 2016 (5,000, no label) | PD scores and drift (PSI) |

**Tuning: expanding-window cross-validation by quarter** inside 2013–14. Each fold trains on all quarters before a validation quarter (2014 Q1, Q2, Q3, Q4), so a model is never validated on loans older than its training data. Hyperparameters are chosen on the average fold result; the calibrator is fitted on the out-of-fold predictions; the final model is refitted on all of 2013–14. A random split is not used, because it would let later loans inform predictions about earlier ones.

**Caveat:** 2014–2015 60-month labels are biased toward early defaults (loans still in progress, see findings A2). Results are also reported by term and by quarter.

## Metrics

- **Ranking:** ROC-AUC, Gini, KS.
- **Probabilities:** calibration plot and Brier score (PDs are compared with 12.5%).
- **Business:** expected profit, approval rate and default rate among approved at the chosen thresholds.
- **Stability:** AUC and calibration by quarter; PSI of PD and top inputs on the holdout.
- **Benchmark:** grade alone (2015 AUC 0.703). Accuracy is not used: with ~17% defaults, approving everyone is 83% "accurate".

## Models

1. **Benchmark:** grade alone.
2. **Scorecard:** logistic regression on the contract's WoE bins; one field per near-duplicate pair (VIF, findings A6).
3. **Gradient boosting (XGBoost)** with monotonic constraints from the findings A7 decile charts: higher utilization, credit-seeking, DTI, loan-to-income and inquiries never lower PD; higher FICO, income and debt size never raise it.
4. **Calibration** (Platt or isotonic, chosen on out-of-fold predictions).

**Class imbalance (~17%)** is mild: no resampling (it distorts probabilities); class weighting is tested as a tuning option, and calibration corrects the probabilities.

**Choice rule, set in advance:** the scorecard is the production model unless XGBoost's 2015 AUC exceeds it by at least **0.01**. Below that gain, the transparent model wins.

**Ablation (required):** raw features only vs factor scores only vs both, reported in the model card.

**grade and int_rate are not inputs.** They are the platform's own risk model; ours must assess risk independently, and pricing follows the risk decision. `installment` is excluded for the same reason (it encodes int_rate). Both serve as the benchmark.

## Decision policy

- Three bands: auto-approve below t1, auto-decline above t2, manual review between. t1 and t2 are chosen by expected profit on 2015; approval and default rates are reported for each band.
- Always manual review: joint applications, FICO below 662, DTI above the training range (outside the approval policy the data comes from).
- Outcomes compared across state regions and income bands.

## Constraints

- **Explainable:** every decision returns 3–4 reason codes in plain words (the features pushing PD up most: scorecard points or XGBoost feature contributions).
- **Latency under 100 ms per application:** measured as p50 / p95 over the full path (validation, transform, model).
- **Reproducible:** settings in a config file, fixed random seeds, every experiment logged to a CSV. The saved model carries its git commit, training-data hash, training date, contract version and thresholds. Inputs are validated against `data_spec.json` after cleaning (fail fast); a model with a different contract version is refused.

## Production (beyond this assessment)

Serve `predict.py` behind an API. Monitor weekly PSI of inputs and PD, approval and review rates, and realized defaults as loans mature. Retrain when PSI exceeds 0.25, performance drops, or yearly, with a refreshed contract. Approve a small controlled sample of borderline applicants to learn about the population the old policy declined.

## Priorities

Work is ordered by rubric weight: model development, pipeline and lineage, evaluation, then error analysis, inference and tests, model card. The Hugging Face demo comes last and may follow the submission.

## What changed

*(to be completed at the end)*