from __future__ import annotations

import json
import os
from typing import Any

from flask import Flask, render_template, request

from data import CROPS, IDEAL, STAGE_LABELS_AR
from ml_predict_supervised import predict_fertilizer
from rules_engine import (
    apply_soil_schedule,
    calc_deficits,
    choose_fertilizer,
    compute_amount_details,
    decision_reason,
    ph_gate,
)

app = Flask(__name__)

CROP_LABELS_AR = {"cucumber": "خيار", "tomato": "بندورة"}


def load_metrics() -> dict[str, Any] | None:
    metrics_path = os.path.join(os.path.dirname(__file__), "model_metrics.json")
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


METRICS = load_metrics()


def _safe_crop(value: str) -> str:
    return value if value in IDEAL else "cucumber"


def _get_stages_for_crop(crop: str) -> list[str]:
    return list(IDEAL[crop].keys())


def _parse_float(form: Any, name: str, default: float = 0.0) -> float:
    raw = form.get(name, "")
    if raw is None or raw == "":
        return default
    return float(raw)


def _stages_by_crop() -> dict[str, list[str]]:
    return {c: list(IDEAL[c].keys()) for c in IDEAL.keys()}


def _base_context(
    *,
    crop: str,
    stages: list[str],
    stage: str,
    result: dict[str, Any] | None = None,
    warning: str | None = None,
    ph_status: str | None = None,
    ph_value: float | None = None,
) -> dict[str, Any]:
    return dict(
        crops=CROPS,
        crop=crop,
        crop_labels=CROP_LABELS_AR,
        stages=stages,
        stage=stage,
        stage_labels=STAGE_LABELS_AR,
        result=result,
        warning=warning,
        ph_status=ph_status,
        ph_value=ph_value,
        metrics=METRICS,
        stages_by_crop=_stages_by_crop(),
    )


def _validate_ph(ph_value: float) -> str | None:
    if not (0 <= ph_value <= 14):
        return "⚠️ قيمة pH يجب أن تكون بين 0 و 14."
    return None


def recommend_from_inputs(
    *,
    crop: str,
    stage: str,
    soilN: float,
    soilP: float,
    soilK: float,
    ph_value: float,
    soil_type: str,
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    ph_status = ph_gate(crop, ph_value)

    if ph_status in ["low", "high"]:
        warning = (
            "⚠️ قيمة pH غير مناسبة حاليًا، التسميد لن يكون فعّالًا قبل تعديل pH "
            "من خلال إضافات محسّنة للتربة (مثل الجير الزراعي أو الكبريت حسب الحالة)."
        )
        return None, warning, ph_status

    warning: str | None = None
    if ph_status == "suboptimal":
        warning = " قيمة pH ليست مثالية."

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
        return result, warning, ph_status

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
    return result, warning, ph_status


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/loading/<target>")
def loading(target: str):
    if target not in ["index", "about"]:
        target = "index"
    return render_template("loading.html", target=target)


@app.route("/index", methods=["GET", "POST"])
def index():
    crop = _safe_crop(request.form.get("crop", "cucumber"))
    stages = _get_stages_for_crop(crop)
    stage = request.form.get("stage", stages[0] if stages else "")

    result: dict[str, Any] | None = None
    warning: str | None = None
    ph_status: str | None = None
    ph_value: float | None = None

    action = request.form.get("action")

    if request.method == "POST" and action != "recommend":
        if stage not in IDEAL[crop]:
            stage = stages[0] if stages else stage
        return render_template("index.html", **_base_context(crop=crop, stages=stages, stage=stage))

    if request.method == "POST" and action == "recommend":
        if stage not in IDEAL[crop]:
            stage = stages[0] if stages else stage

        try:
            soilN = _parse_float(request.form, "soilN")
            soilP = _parse_float(request.form, "soilP")
            soilK = _parse_float(request.form, "soilK")
            ph_value = _parse_float(request.form, "ph")
        except ValueError:
            warning = "⚠️ تأكدي إن القيم أرقام صحيحة."
            return render_template(
                "index.html",
                **_base_context(crop=crop, stages=stages, stage=stage, warning=warning),
            )

        ph_warning = _validate_ph(ph_value)
        if ph_warning:
            return render_template(
                "index.html",
                **_base_context(
                    crop=crop,
                    stages=stages,
                    stage=stage,
                    warning=ph_warning,
                    ph_value=ph_value,
                ),
            )

        soil_type = request.form.get("soil_type") or "loam"

        result, warning, ph_status = recommend_from_inputs(
            crop=crop,
            stage=stage,
            soilN=soilN,
            soilP=soilP,
            soilK=soilK,
            ph_value=ph_value,
            soil_type=soil_type,
        )

    return render_template(
        "index.html",
        **_base_context(
            crop=crop,
            stages=stages,
            stage=stage,
            result=result,
            warning=warning,
            ph_status=ph_status,
            ph_value=ph_value,
        ),
    )


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug)
