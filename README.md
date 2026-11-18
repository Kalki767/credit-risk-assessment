# Credit risk assessment: PD model on a data contract

Part A (data scientist) cleans three messy sources and packages a data contract; Part B (ML engineer) builds, evaluates and serves a probability-of-default model on that contract.

## Setup
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf download metasebiya21/credit-risk-take-home --repo-type dataset --local-dir data/raw   # data is never committed
```

## How to run
| Step | Command |
| --- | --- |
| Part A analysis | notebooks 01–08 in order |
| Build the contract | [contract build notebook/script] → `contract/` |
| Check contract parity (50 rows, 1e-6) | `python scripts/check_parity.py` |
| Train (one command: validate, CV, evaluate, thresholds, save model + metadata, log) | `python scripts/train.py` |
| Model analysis (evaluation, thresholds, error analysis) | `notebooks/09_model.ipynb` (sections B1–B17) |
| Score the 2016 holdout + drift report | `python scripts/score_holdout.py` → `reports/holdout_scores.csv`, `reports/drift_holdout.csv` |
| Score one application | `python -m src.predict examples/high_risk.json` |
| Tests | `python -m pytest -q` |
| Latency benchmark | `python scripts/benchmark_latency.py` → `reports/latency.json` |
| Demo app (Gradio) | `python scripts/build_space.py && cd build/space && python app.py` → http://127.0.0.1:7860 |

## Deliverables
DESIGN.md · model_card.md · contract/ (findings.md inside) · experiments/log.csv · tests/ · reports/ (holdout scores, drift, latency, charts) · space/ (Hugging Face app).

**Hugging Face Space:** creating a Gradio Space on CPU now returns HTTP 402 ("requires a PRO subscription") for free accounts. The Space is fully built and deploys with one command (`python scripts/build_space.py --deploy USER/NAME`) on a paid account; the build refuses any data-like file. It is demoed live in the interview.

## Time spent
| Date | Hours | Work |
| --- | --- | --- |
| Oct 30 | 3.5 | setup, data audit |
| Oct 31 | 5.5 | A1–A3 |
| Nov 1 | 6.5 | A3–A5 |
| Nov 2 | 7 | A6–A8, contract, findings.md |
| Nov 3 | [ ] | DESIGN.md, pipeline, scorecard, XGBoost, calibration, ablation, final test |
| Nov 4 | [ ] | thresholds, evaluation, error analysis, train.py, holdout, predict.py, tests, demo, docs |

## AI tools used
- **Claude (Anthropic):** planning the work against the brief; explaining concepts (WoE/IV, PCA, factor analysis, PSI, calibration, AUC/KS) before I implemented them; drafting code as small functions, which I ran, checked against expected outputs and committed step by step; reviewing my results for mistakes (it found, for example, a label leak in the scorecard's cross-validation); drafting documentation (findings.md, DESIGN.md, model card, README), which I reviewed and edited. Every decision and judgement call is recorded in the notebooks' markdown and the model card.
