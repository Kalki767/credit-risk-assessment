---
title: Credit Risk PD Scorecard
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
pinned: false
---

# Credit risk PD scorecard (demo)

Scores one loan application: probability of default (PD), decision band and reason codes.
The app calls the same `predict()` used by the project's CLI and tests. **All example applicants are made up; no real data is included.**

## Model card (short)

- **Model:** WoE scorecard (logistic regression on 14 weight-of-evidence features from the data contract), plus a 2-parameter recalibration. Monotone, so ranking and reason codes come straight from the scorecard.
- **Data window:** trained on loans issued 2013–2014; evaluated once on 2015 loans; recalibrated on 2015 outcomes after the evaluation (a conservative production refresh).
- **Metrics on 2015 (unseen):** AUC 0.7125 (Gini 0.425, KS 0.309) vs 0.7026 for the platform's grade alone; stable by quarter (AUC 0.69–0.72).
- **Thresholds:** repaid loans earn 10% of the amount, defaults lose 70%, so the break-even PD is 12.5%. Auto-approve below 11.3%, manual review 11.3–13.8%, auto-decline from 13.8%. Joint applications, FICO below the training range and DTI above it go to review.
- **Not used:** grade, sub_grade, int_rate, installment (the old model's outputs), location (fairness), and anything recorded after the loan was issued.
- **Known limits:**
  - trained on accepted loans only
  - recent and 60-month loans have biased labels (loans still running were removed), so 60-month PDs are less reliable
  - high incomes are over-credited
  - built on US Lending Club loans from 2013–2015