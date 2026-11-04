"""Hugging Face Space: score a made-up applicant with the SAME predict() used by the CLI and the tests."""
import json
from functools import partial
from pathlib import Path

import gradio as gr

from src.predict import artifacts, predict

ROOT = Path(__file__).resolve().parent
_, META, _, SPEC = artifacts()
FIELDS = SPEC["fields"]
EXAMPLES = {name: json.load(open(ROOT / "examples" / f"{name}.json"))
            for name in ("low_risk", "borderline", "high_risk")}


def make_input(f):
    """One form input per data_spec field: numbers show their valid range, categories their allowed values."""
    if f["type"] in ("float", "int"):
        lo, hi = f.get("min"), f.get("max")
        info = f"valid range: {lo if lo is not None else '-'} to {hi if hi is not None else '-'}"
        return gr.Number(label=f["name"], info=info)
    if f.get("allowed_values"):
        return gr.Dropdown(choices=[str(v) for v in f["allowed_values"]], label=f["name"], allow_custom_value=True)
    return gr.Textbox(label=f["name"])


def score(*values):
    """Form values -> predict() -> a readable summary + the full JSON output (or a clear error)."""
    application = {f["name"]: (None if v in ("", None) else v) for f, v in zip(FIELDS, values)}
    try:
        out = predict(application)
    except ValueError as e:
        return f"### Input rejected\n\n```\n{e}\n```", {}
    reasons = "\n".join(f"{i}. {r}" for i, r in enumerate(out["reason_codes"], 1)) or "none"
    t = out["metadata"]["thresholds"]
    summary = (f"## PD {out['pd']:.1%} → {out['decision']}\n\n**Why:** {out['decision_basis']}\n\n"
               f"Auto-approve below {t['auto_approve_below']:.1%} · review · auto-decline from "
               f"{t['auto_decline_from']:.1%} (break-even {t['break_even_pd']:.1%})\n\n"
               f"**Reason codes** ({out['reason_codes_meaning']})\n\n{reasons}\n\n"
               f"**Review triggers:** {', '.join(out['review_triggers']) or 'none'} · "
               f"lowest possible PD {out['pd_lowest_possible']:.1%}\n\n"
               f"Model {out['model_version']} · trained {out['metadata']['training_date'][:10]} · "
               f"commit {out['metadata']['git_commit'][:8]} · contract {out['metadata']['contract_version']}")
    return summary, out


def load_example(name):
    return [EXAMPLES[name].get(f["name"]) for f in FIELDS]


with gr.Blocks(title="Credit risk PD scorecard") as demo:
    gr.Markdown("# Credit risk PD scorecard\n"
                "Probability of default at application time, the decision band and the reason codes. "
                f"Model {META['model_version']}, trained {META['training_date'][:10]}, "
                f"commit {META['git_commit'][:8]}. **The example applicants are made up.**")
    with gr.Row():
        buttons = {name: gr.Button(f"Example: {name.replace('_', ' ')}") for name in EXAMPLES}
    inputs = []
    half = (len(FIELDS) + 1) // 2
    with gr.Row():
        for part in (FIELDS[:half], FIELDS[half:]):
            with gr.Column():
                inputs += [make_input(f) for f in part]
    run = gr.Button("Score applicant", variant="primary")
    summary = gr.Markdown()
    full = gr.JSON(label="Full predict() output")
    run.click(score, inputs, [summary, full])
    for name, button in buttons.items():
        button.click(partial(load_example, name), None, inputs).then(score, inputs, [summary, full])

if __name__ == "__main__":
    demo.launch()