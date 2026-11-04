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

**Tuning: expanding-window cross-validation by quarter** inside 2013–14. Each fold trains on every quarter before its validation quarter, so a model is never validated on loans older than its training data:

| Fold | Train | Train loans | Validate | Validation loans |
| --- | --- | --- | --- | --- |
| 1 | 2013 Q1–Q4 | 5,349 | 2014 Q1 | 1,906 |
| 2 | 2013 Q1 – 2014 Q1 | 7,255 | 2014 Q2 | 2,091 |
| 3 | 2013 Q1 – 2014 Q2 | 9,346 | 2014 Q3 | 2,336 |
| 4 | 2013 Q1 – 2014 Q3 | 11,682 | 2014 Q4 | 2,691 |

2013 is used for training in every fold. Validation starts in 2014 so every fold trains on at least a full year; validating on 2013 Q2 or Q3 would mean training on only 893 or 2,139 loans, giving noisy, pessimistic scores. Hyperparameters are chosen on the average fold AUC; the calibrator is fitted on the out-of-fold predictions; the final model is refitted on all of 2013–14. A random split is not used, because it would let later loans inform predictions about earlier ones.

**Caveat:** 2014–2015 60-month labels are biased toward early defaults (loans still in progress, see findings A2). Results are also reported by term and by quarter.

## Metrics, in priority order

| Priority | Question | Metric | Used for |
| --- | --- | --- | --- |
| 1 | Does it rank borrowers correctly? | ROC-AUC (Gini = 2·AUC − 1) | tuning hyperparameters; choosing the model |
| 2 | Are its probabilities honest? | calibration plot, Brier score | choosing the calibration method; makes the 12.5% threshold meaningful |
| 3 | Does it make money? | expected profit, approval rate, default rate among approved | setting the thresholds |

Ranking comes first because calibration can be corrected afterwards, while poor ranking cannot. Also reported: **KS** (the industry's traditional separation measure), **AUC and calibration by quarter** (stability), **grade alone** as the benchmark to beat (2015 AUC 0.703), and PSI of PD and top inputs on the holdout. Accuracy is not used: with ~17% defaults, approving everyone is 83% "accurate".

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

## Production (out of scope, in brief)
- **Serve:** the same `predict()` behind a small internal API or a batch job; the model and transform load once at start-up (p50 latency measured in reports/latency.json); invalid applications are rejected with the validation message.
- **Monitor:** weekly PSI of the PD and the top inputs (investigate at 0.10, act at 0.25), the decision mix and validation-failure rate, and, most importantly, predicted vs realized default rate per monthly vintage as loans mature.
- **Retrain:** recalibrate quarterly on matured loans; retrain with `scripts/train.py` when the vintage gap persists or PSI exceeds 0.25; a contract change (new features) gets a new contract_version, and the new model runs as a challenger before replacing the old one.

## Priorities

Work is ordered by rubric weight: model development, pipeline and lineage, evaluation, then error analysis, inference and tests, model card. The Hugging Face demo comes last and may follow the submission.

## What changed (updated at the end)

- **WoE re-learned inside each CV fold:** the contract's WoE values used labels from the validation quarters (CV AUC 0.7028 → 0.6975 after the fix).
- **XGBoost ended at depth 1** (interactions did not help); monotonic constraints improved CV AUC. It still lost the 0.01 rule, so the **scorecard is the production model**, as planned.
- **Factor scores not used** in the final model (ablation +0.0019, below the 0.002 rule).
- **Thresholds:** the cut-off is chosen on 2014 out-of-fold PDs and tested on 2015; three bands with an assumed 10% review capacity; review triggers for joint applications and FICO/DTI outside the training range.
- **Production recalibration (deviation from the planned split):** after the 2015 evaluation was frozen, a 2-parameter recalibration was learned on 2015 outcomes, because PDs ran about 3 points low and the credit team prefers conservative PDs. It is monotone (AUC and reason codes unchanged); its 2015 results are in-sample and reported as a check only.
- **Demo hosting:** Hugging Face now requires a paid plan for Gradio Spaces on CPU (HTTP 402 when creating the Space). The Space is fully built (`space/`, `scripts/build_space.py`, one-command deploy) and is demoed live from a laptop.
- **Latency target missed:** p50 149 ms / p95 221 ms (target 100 ms) on a 2017 laptop. The per-row pandas transform has a fixed cost per operation; the planned fix is a lean serving path for the 14 features, verified by the parity test (reports/latency.json).