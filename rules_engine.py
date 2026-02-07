"""
Rule-based (expert) engine for fertilizer recommendation.

This module contains pure functions to:
- Compute N/P/K deficits from soil values vs ideal targets
- Gate recommendations based on pH ranges
- Choose fertilizer type based on dominant deficit
- Compute fertilizer amount from fertilizer composition
- Generate an application schedule based on soil type
- Produce a full expert recommendation dictionary
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any

from data import IDEAL, PH_RANGES

FERTILIZERS: Dict[str, Dict[str, float]] = {
    "NPK_20_20_20": {"N": 0.20, "P": 0.20, "K": 0.20},
    "UREA_46_0_0": {"N": 0.46, "P": 0.00, "K": 0.00},
    "MAP_12_61_0": {"N": 0.12, "P": 0.61, "K": 0.00},
    "KNO3_13_0_46": {"N": 0.13, "P": 0.00, "K": 0.46},
}

SOIL_INTERVAL_DAYS: Dict[str, int] = {
    "sandy": 2,
    "clay": 3,
}

SOIL_AR: Dict[str, str] = {
    "sandy": "رملية",
    "clay": "طينية",
}


def calc_deficits(
    crop: str,
    stage: str,
    soilN: float,
    soilP: float,
    soilK: float,
) -> Tuple[dict, float, float, float]:
    """
    Calculate nutrient deficits relative to ideal targets.

    Parameters
    ----------
    crop : str
        Crop key.
    stage : str
        Stage key for the crop.
    soilN : float
        Current soil nitrogen value.
    soilP : float
        Current soil phosphorus value.
    soilK : float
        Current soil potassium value.

    Returns
    -------
    tuple
        (ideal_targets_dict, defN, defP, defK)
    """
    ideal = IDEAL[crop][stage]
    defN = max(0.0, float(ideal["N"]) - float(soilN))
    defP = max(0.0, float(ideal["P"]) - float(soilP))
    defK = max(0.0, float(ideal["K"]) - float(soilK))
    return ideal, defN, defP, defK


def ph_gate(crop: str, ph_value: float) -> str:
    """
    Classify pH status for a given crop.

    Returns one of: 'low', 'high', 'optimal', 'suboptimal'.

    Parameters
    ----------
    crop : str
        Crop key.
    ph_value : float
        Soil pH value.

    Returns
    -------
    str
        pH status category.
    """
    rng = PH_RANGES.get(crop, {"optimal": (6.0, 6.8), "suboptimal": (5.5, 7.5)})
    opt_lo, opt_hi = rng["optimal"]
    sub_lo, sub_hi = rng["suboptimal"]
    if ph_value < sub_lo:
        return "low"
    if ph_value > sub_hi:
        return "high"
    if opt_lo <= ph_value <= opt_hi:
        return "optimal"
    return "suboptimal"


def choose_fertilizer(defN: float, defP: float, defK: float) -> str:
    """
    Choose an appropriate fertilizer name based on the dominant deficit.

    Parameters
    ----------
    defN, defP, defK : float
        Nutrient deficits.

    Returns
    -------
    str
        Fertilizer key from FERTILIZERS.
    """
    dN, dP, dK = float(defN), float(defP), float(defK)
    total = dN + dP + dK
    if total <= 0:
        return "NPK_20_20_20"

    ratios = {"N": dN, "P": dP, "K": dK}
    top = max(ratios, key=ratios.get)

    if ratios[top] / total >= 0.5:
        if top == "N":
            return "UREA_46_0_0"
        if top == "P":
            return "MAP_12_61_0"
        return "KNO3_13_0_46"

    return "NPK_20_20_20"


def _target_for_amount(
    fert_name: str,
    defN: float,
    defP: float,
    defK: float,
) -> Tuple[str, float, float]:
    """
    Decide which nutrient deficit to target for calculating fertilizer amount.

    Returns
    -------
    tuple
        (target_nutrient, target_deficit, fertilizer_fraction)
    """
    comp = FERTILIZERS[fert_name]
    dN, dP, dK = float(defN), float(defP), float(defK)

    if fert_name == "UREA_46_0_0":
        return "N", dN, float(comp["N"])
    if fert_name == "MAP_12_61_0":
        return "P", dP, float(comp["P"])
    if fert_name == "KNO3_13_0_46":
        return "K", dK, float(comp["K"])

    # NPK: target the largest deficit among N/P/K
    if dK >= dN and dK >= dP:
        return "K", dK, float(comp["K"])
    if dN >= dP:
        return "N", dN, float(comp["N"])
    return "P", dP, float(comp["P"])


def compute_amount_details(
    fert_name: str,
    defN: float,
    defP: float,
    defK: float,
) -> Dict[str, Any]:
    """
    Compute fertilizer amount (g/dunum/day) based on nutrient deficit and composition.

    Parameters
    ----------
    fert_name : str
        Fertilizer key.
    defN, defP, defK : float
        Nutrient deficits.

    Returns
    -------
    dict
        Includes target nutrient, deficit, fraction, amount_g, and equation text.
    """
    target, deficit, frac = _target_for_amount(fert_name, defN, defP, defK)
    if deficit <= 0 or frac <= 0:
        return {
            "target": target,
            "deficit": round(float(deficit), 2),
            "fraction": round(float(frac), 4),
            "amount_g": 0.0,
            "equation": "amount = 0",
        }

    amount = float(deficit) / float(frac)
    return {
        "target": target,
        "deficit": round(float(deficit), 2),
        "fraction": round(float(frac), 4),
        "amount_g": round(float(amount), 2),
        "equation": f"amount = deficit / fraction = {round(float(deficit), 2)} / {round(float(frac), 4)}",
    }


def decision_reason(defN: float, defP: float, defK: float) -> str:
    """
    Provide a short Arabic explanation for why a fertilizer choice was made.

    Returns
    -------
    str
        Reason text.
    """
    deficits = {"N": float(defN), "P": float(defP), "K": float(defK)}
    main = max(deficits, key=deficits.get)

    if deficits[main] <= 0:
        return "لا يوجد نقص فعلي، التسميد غير ضروري الآن."
    if main == "N":
        return "أكبر نقص في النيتروجين (N)، لذلك نميل لسماد غني بالـ N."
    if main == "P":
        return "أكبر نقص في الفوسفور (P)، لذلك نميل لسماد عالي الفوسفور."
    return "أكبر نقص في البوتاسيوم (K)، لذلك نميل لسماد غني بالبوتاسيوم."


def apply_soil_schedule(daily_amount_g: float, soil_type: Optional[str]) -> Dict[str, Any]:
    """
    Convert daily fertilizer amount to an application schedule based on soil type.

    Parameters
    ----------
    daily_amount_g : float
        Daily recommended amount (g/dunum/day).
    soil_type : str | None
        Soil type key (e.g., 'sandy' or 'clay').

    Returns
    -------
    dict
        Contains soil_type, interval_days, dose_per_application, and recommendation text.
    """
    soil_key = (soil_type or "").strip().lower()
    interval_days = SOIL_INTERVAL_DAYS.get(soil_key)
    if interval_days is None:
        interval_days = 2
        soil_key = "sandy"

    soil_ar = SOIL_AR.get(soil_key, soil_key)

    if float(daily_amount_g) <= 0:
        return {
            "soil_type": soil_key,
            "interval_days": interval_days,
            "dose_per_application_g_per_dunum": 0.0,
            "schedule_recommendation": (
                f"نوع التربة: {soil_ar}. التربة جيدة ولا يوجد نقص فعلي، "
                "لا حاجة لتسميد كيميائي. يمكن إضافة سماد عضوي فقط للتحسين."
            ),
        }

    dose_per_application = float(daily_amount_g) * float(interval_days)
    recommendation = (
        f"نوع التربة: {soil_ar}. "
        f"أضف {round(dose_per_application, 2)} g/dunum "
        f"كل {interval_days} يوم/أيام مع ماء الري "
        f"(الكمية اليومية المحسوبة {round(float(daily_amount_g), 2)} g/dunum/day)."
    )

    return {
        "soil_type": soil_key,
        "interval_days": interval_days,
        "dose_per_application_g_per_dunum": round(dose_per_application, 2),
        "schedule_recommendation": recommendation,
    }


def expert_recommendation(
    crop: str,
    stage: str,
    soilN: float,
    soilP: float,
    soilK: float,
    ph: Optional[float] = None,
    soil_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a full expert recommendation using rule-based logic.

    Parameters
    ----------
    crop : str
        Crop key.
    stage : str
        Stage key.
    soilN, soilP, soilK : float
        Soil nutrient values.
    ph : float, optional
        Soil pH (passed through in output), by default None.
    soil_type : str, optional
        Soil type key for scheduling, by default None.

    Returns
    -------
    dict
        Recommendation dictionary including deficits, fertilizer type/amount, and schedule.
    """
    ideal, defN, defP, defK = calc_deficits(crop, stage, soilN, soilP, soilK)
    fert = choose_fertilizer(defN, defP, defK)
    details = compute_amount_details(fert, defN, defP, defK)

    daily_amount = float(details["amount_g"])
    schedule = apply_soil_schedule(daily_amount, soil_type)

    return {
        "crop": crop,
        "stage": stage,
        "ideal": ideal,
        "defN": round(float(defN), 2),
        "defP": round(float(defP), 2),
        "defK": round(float(defK), 2),
        "fertilizer_type": fert,
        "fertilizer_amount_g": daily_amount,
        "unit": "g/dunum/day",
        "soil_type": schedule["soil_type"],
        "interval_days": schedule["interval_days"],
        "dose_per_application_g_per_dunum": schedule["dose_per_application_g_per_dunum"],
        "schedule_recommendation": schedule["schedule_recommendation"],
        "amount_details": details,
        "ph": ph,
    }
