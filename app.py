"""
Flask web application for fertilizer recommendation.

This module provides the web UI routes and helper functions to:
- Collect soil/crop inputs from the user
- Validate inputs (including pH checks)
- Compute NPK deficits and recommendation logic
- Use ML prediction (if available) and expert rule-based recommendation
- Render results using HTML templates
"""

from __future__ import annotations

from flask import Flask, render_template, request
import os
import json

from data import CROPS, IDEAL, STAGE_LABELS_AR
from rules_engine import (
    calc_deficits,
    ph_gate,
    decision_reason,
    compute_amount_details,
    choose_fertilizer,
    apply_soil_schedule,
)
from ml_predict_supervised import predict_fertilizer

app = Flask(__name__)

CROP_LABELS_AR = {"cucumber": "خيار", "tomato": "بندورة"}

METRICS = None
metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            METRICS = json.load(f)
    except Exception:
        METRICS = None


def _safe_crop(value: str) -> str:
    """
    Return a valid crop key. If the provided value is not supported, default to 'cucumber'.

    Parameters
    ----------
    value : str
        Crop key from the form input.

    Returns
    -------
    str
        A valid crop key present in IDEAL.
    """
    return value if value in IDEAL else "cucumber"


def _get_stages_for_crop(crop: str) -> list:
    """
    Get stage keys for a given crop.

    Parameters
    ----------
    crop : str
        Crop key.

    Returns
    -------
    list
        List of stage keys.
    """
    return list(IDEAL[crop].keys())


def _parse_float(name: str, default: float = 0.0) -> float:
    """
    Parse a float from a Flask form field.

    Parameters
    ----------
    name : str
        Field name in request.form.
    default : float, optional
        Value to return if the field is missing/empty, by default 0.0.

    Returns
    -------
    float
        Parsed numeric value.

    Raises
    ------
    ValueError
        If the field exists but cannot be converted to float.
    """
    raw = request.form.get(name, "")
    if raw is None or raw == "":
        return default
    return float(raw)


@app.route("/")
def landing():
    """
    Render the landing page.

    Returns
    -------
    str
        Rendered HTML.
    """
    return render_template("landing.html")


@app.route("/about")
def about():
    """
    Render the about page.

    Returns
    -------
    str
        Rendered HTML.
    """
    return render_template("about.html")


@app.route("/loading/<target>")
def loading(target: str):
    """
    Render a loading page that redirects to a target page.

    Parameters
    ----------
    target : str
        Target page name (e.g., 'index' or 'about').

    Returns
    -------
    str
        Rendered HTML.
    """
    if target not in ["index", "about"]:
        target = "index"
    return render_template("loading.html", target=target)


@app.route("/index", methods=["GET", "POST"])
def index():
    """
    Main recommendation route.

    - GET: renders the form with defaults.
    - POST (action != recommend): resets the form state for the chosen crop/stage.
    - POST (action == recommend): validates inputs, checks pH, computes deficits, and returns recommendation.

    Returns
    -------
    str
        Rendered HTML for the index page.
    """
    crop = _safe_crop(request.form.get("crop", "cucumber"))
    stages = _get_stages_for_crop(crop)
    stage = request.form.get("stage", stages[0] if stages else "")

    result = None
    warning = None
    ph_status = None
    ph_value = None

    action = request.form.get("action")

    if request.method == "POST" and action != "recommend":
        if stage not in IDEAL[crop]:
            stage = stages[0] if stages else stage

        return render_template(
            "index.html",
            crops=CROPS,
            crop=crop,
            crop_labels=CROP_LABELS_AR,
            stages=stages,
            stage_labels=STAGE_LABELS_AR,
            result=None,
            warning=None,
            ph_status=None,
            ph_value=None,
            metrics=METRICS,
            stages_by_crop={c: list(IDEAL[c].keys()) for c in IDEAL.keys()},
        )

    if request.method == "POST" and action == "recommend":
        if stage not in IDEAL[crop]:
            stage = stages[0] if stages else stage

        try:
            soilN = _parse_float("soilN")
            soilP = _parse_float("soilP")
            soilK = _parse_float("soilK")
            ph_value = _parse_float("ph")
        except ValueError:
            warning = "⚠️ تأكدي إن القيم أرقام صحيحة."
            return render_template(
                "index.html",
                crops=CROPS,
                crop=crop,
                crop_labels=CROP_LABELS_AR,
                stages=stages,
                stage_labels=STAGE_LABELS_AR,
                result=None,
                warning=warning,
                ph_status=None,
                ph_value=None,
                metrics=METRICS,
                stages_by_crop={c: list(IDEAL[c].keys()) for c in IDEAL.keys()},
            )

        if not (0 <= ph_value <= 14):
            warning = "⚠️ قيمة pH يجب أن تكون بين 0 و 14."
            return render_template(
                "index.html",
                crops=CROPS,
                crop=crop,
                crop_labels=CROP_LABELS_AR,
                stages=stages,
                stage_labels=STAGE_LABELS_AR,
                result=None,
                warning=warning,
                ph_status=None,
                ph_value=ph_value,
                metrics=METRICS,
                stages_by_crop={c: list(IDEAL[c].keys()) for c in IDEAL.keys()},
            )

        ph_status = ph_gate(crop, ph_value)

        if ph_status in ["low", "high"]:
            warning = (
                "⚠️ قيمة pH غير مناسبة حاليًا، التسميد لن يكون فعّالًا قبل تعديل pH "
                "من خلال إضافات محسّنة للتربة (مثل الجير الزراعي أو الكبريت حسب الحالة)."
            )

            return render_template(
                "index.html",
                crops=CROPS,
                crop=crop,
                crop_labels=CROP_LABELS_AR,
                stages=stages,
                stage_labels=STAGE_LABELS_AR,
                result=None,
                warning=warning,
                ph_status=ph_status,
                ph_value=ph_value,
                metrics=METRICS,
                stages_by_crop={c: list(IDEAL[c].keys()) for c in IDEAL.keys()},
            )

        if ph_status == "suboptimal":
            warning = " قيمة pH ليست مثالية."

        soil_type = request.form.get("soil_type")

        ideal, defN, defP, defK = calc_deficits(crop, stage, soilN, soilP, soilK)

        total_def = float(defN) + float(defP) + float(defK)

        if total_def <= 0:
            schedule = apply_soil_schedule(0.0, soil_type)
            warning = (
                "✅ التربة جيدة ولا يوجد نقص في NPK. يمكن إضافة سماد عضوي فقط للتحسين "
                "(كمبوست/سماد بلدي) مع ماء الري."
            )
            result = {
                "crop": crop,
                "stage": stage,
                "ideal": ideal,
                "defN": round(defN, 2),
                "defP": round(defP, 2),
                "defK": round(defK, 2),
                "fertilizer_type": "—",
                "fertilizer_amount_g": 0.0,
                "ml_confidence": None,
                "expert_type": "NPK_20_20_20",
                "expert_amount_g": 0.0,
                "amount_details": {
                    "target": "—",
                    "deficit": 0.0,
                    "fraction": 0.0,
                    "amount_g": 0.0,
                    "equation": "amount = 0",
                },
                "reason": "لا يوجد نقص فعلي، التسميد الكيميائي غير ضروري الآن.",
                "soil_type": schedule["soil_type"],
                "interval_days": schedule["interval_days"],
                "dose_per_application_g_per_dunum": schedule["dose_per_application_g_per_dunum"],
                "schedule_recommendation": schedule["schedule_recommendation"],
                "unit": "g/dunum/day",
            }
            return render_template(
                "index.html",
                crops=CROPS,
                crop=crop,
                crop_labels=CROP_LABELS_AR,
                stages=stages,
                stage_labels=STAGE_LABELS_AR,
                result=result,
                warning=warning,
                ph_status=ph_status,
                ph_value=ph_value,
                metrics=METRICS,
                stages_by_crop={c: list(IDEAL[c].keys()) for c in IDEAL.keys()},
            )

        fert_type, amount_ml, confidence = predict_fertilizer(
            crop=crop,
            stage=stage,
            soilN=soilN,
            soilP=soilP,
            soilK=soilK,
            ph=ph_value,
            defN=defN,
            defP=defP,
            defK=defK,
        )

        expert_type = choose_fertilizer(defN, defP, defK)
        expert_details = compute_amount_details(expert_type, defN, defP, defK)

        schedule = apply_soil_schedule(float(amount_ml), soil_type)
        reason_text = decision_reason(defN, defP, defK)

        result = {
            "crop": crop,
            "stage": stage,
            "ideal": ideal,
            "defN": round(defN, 2),
            "defP": round(defP, 2),
            "defK": round(defK, 2),
            "fertilizer_type": fert_type,
            "fertilizer_amount_g": round(float(amount_ml), 2),
            "ml_confidence": confidence,
            "expert_type": expert_type,
            "expert_amount_g": round(float(expert_details.get("amount_g", 0.0)), 2),
            "amount_details": expert_details,
            "reason": reason_text,
            "soil_type": schedule["soil_type"],
            "interval_days": schedule["interval_days"],
            "dose_per_application_g_per_dunum": schedule["dose_per_application_g_per_dunum"],
            "schedule_recommendation": schedule["schedule_recommendation"],
            "unit": "g/dunum/day",
        }

    return render_template(
        "index.html",
        crops=CROPS,
        crop=crop,
        crop_labels=CROP_LABELS_AR,
        stages=stages,
        stage_labels=STAGE_LABELS_AR,
        result=result,
        warning=warning,
        ph_status=ph_status,
        ph_value=ph_value,
        metrics=METRICS,
        stages_by_crop={c: list(IDEAL[c].keys()) for c in IDEAL.keys()},
    )


if __name__ == "__main__":
    app.run(debug=True)
