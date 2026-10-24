"""GLIMPSE screen (Step 6b): a small local web page on top of the same `Glimpse.predict` the demo and the
evaluation use. Start with `just app`, then open http://127.0.0.1:7860. Runs offline on the local weights.

The user gives a before-meal photo, the CGM export of the last 2 hours (CSV with timestamp, glucose_mgdl and
optionally hr_bpm) and a short profile. Example meals from the test people can be loaded with one click."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from loguru import logger

from glimpse.config import load_config
from glimpse.data.dataset import MealDataset
from glimpse.data.splits import Splits
from glimpse.demo import Glimpse, answer_from_files
from glimpse.demo_run import write_example
from glimpse.feedback import record_feedback
from glimpse.utils.logging import setup_logging

COLOURS = {"low risk": "#1e8449", "caution": "#b9770e", "can't judge": "#5d6d7e"}
DISCLAIMER = ("Research prototype for a second look before eating. Not a medical device and not medical advice. "
              "Trained on 28 adults (CGMacros); results for other people may differ.")  # fmt: skip


def _examples(cfg) -> dict[str, dict]:
    """The demo's worked cases (chosen by a fixed rule) as ready-made files, if the dataset is available."""
    import json

    summary = Path("outputs/demo/demo_summary.json")
    if not (summary.is_file() and Path("data/dataset.json").is_file()):
        return {}
    ds, _ = MealDataset.load(Path("data/dataset")), Splits.load(Path("splits/dev_splits.json"))
    picks = json.loads(summary.read_text())["cases"]
    return {title: write_example(cfg, ds, row, Path("outputs/demo/examples")) for title, row in picks.items() if row}


def build(cfg=None) -> gr.Blocks:
    """The page: inputs on the left, the answer, examples and the feedback form on the right."""
    cfg = cfg or load_config()
    app = Glimpse(cfg)
    examples = _examples(cfg)

    def answer(photo, csv, meal_time, age, bmi, a1c, fasting, prior_rate, prior_rise):
        """Screen inputs -> the same validated Answer as `glimpse predict` (bad input -> can't judge)."""
        profile = {"age": age, "bmi": bmi, "a1c_pct": a1c, "fasting_glucose_mgdl": fasting,
                   "prior_spike_rate": prior_rate, "prior_rise": prior_rise}  # fmt: skip
        a = answer_from_files(app, csv, photo, meal_time or None, profile)
        state = {"meal_time": meal_time, **a.model_dump()}
        head = f"<h2 style='color:{COLOURS[a.decision]}'>{a.decision.upper()}</h2>\n\n{a.sentence}"
        rows = [f"- **Why:** {a.reason}"]
        if a.carb_bin:
            rows.append(f"- **Carbs on the plate:** most likely {a.carb_bin} ({a.carb_bin_prob:.0%} sure)")
        rows.append(f"- **Answered in** {a.latency_ms or 0:.0f} ms")
        return head, "\n".join(rows), state

    def load_example(title):
        """Fill the inputs with one of the demo's worked meals."""
        e = examples[title]
        p = e["profile"]
        r = [None if v != v else round(float(v), 2) for v in (p["age"], p["bmi"], p["a1c_pct"],
             p["fasting_glucose_mgdl"], p["prior_spike_rate"], p["prior_rise"])]  # fmt: skip
        return (str(e["photo"]) if e["photo"] else None, str(e["csv"]), e["meal_time"].isoformat(), *r)

    with gr.Blocks(title="GLIMPSE") as ui:
        gr.Markdown("# GLIMPSE\nWill this meal spike my glucose? Photo + last 2 h of CGM + a short profile.")
        with gr.Row():
            with gr.Column():
                photo = gr.Image(type="filepath", label="Before-meal photo")
                csv = gr.File(label="CGM export, last 2 h (CSV: timestamp, glucose_mgdl, hr_bpm)", file_types=[".csv"])
                meal_time = gr.Textbox(label="Meal time (YYYY-MM-DDTHH:MM; empty = last reading)")
                with gr.Row():
                    age, bmi = gr.Number(label="Age", value=45), gr.Number(label="BMI", value=28)
                    a1c = gr.Number(label="HbA1c (%)", value=5.9)
                    fasting = gr.Number(label="Fasting glucose", value=100)
                with gr.Row():
                    prior_rate = gr.Number(label="Onboarding spike rate (0-1, optional)")
                    prior_rise = gr.Number(label="Onboarding mean rise (mg/dL, optional)")
                go = gr.Button("Check this meal", variant="primary")
            with gr.Column():
                head, details = gr.Markdown(), gr.Markdown()
                if examples:
                    pick = gr.Dropdown(list(examples), value=None, label="Or load an example meal (test people)")
                    pick.change(load_example, pick, [photo, csv, meal_time, age, bmi, a1c, fasting, prior_rate,
                                                     prior_rise])  # fmt: skip
                with gr.Accordion("After the meal: what happened? (helps improve GLIMPSE)", open=False):
                    ate = gr.Radio(["ate it all", "ate part of it", "did not eat it"], label="Did you eat it?")
                    rise = gr.Number(label="Rise seen on your CGM in the 2 h after (mg/dL, optional)")
                    send, sent = gr.Button("Send feedback"), gr.Markdown()
                gr.Markdown(f"*{DISCLAIMER}*")
        last = gr.State({})
        go.click(answer, [photo, csv, meal_time, age, bmi, a1c, fasting, prior_rate, prior_rise], [head, details, last])

        def give_feedback(state, ate_value, rise_value):
            """Save the user's correction next to what the app said (mocked feedback loop)."""
            if not state:
                return "Check a meal first."
            r = record_feedback(cfg, state, ate_value, rise_value, source="user")
            return f"Thanks, saved ({r.feedback_label})."

        send.click(give_feedback, [last, ate, rise], sent)
    return ui


def main() -> None:
    """Start the local server (offline, local weights)."""
    cfg = load_config()
    setup_logging(cfg.paths.logs)
    logger.bind(stage="app").info("starting the GLIMPSE screen on http://127.0.0.1:7860")
    build(cfg).launch(server_name="127.0.0.1", server_port=7860)


if __name__ == "__main__":
    main()
