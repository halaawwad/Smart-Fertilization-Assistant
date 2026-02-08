"""
Supervised ML prediction utilities for fertilizer recommendation.

This module loads trained models (classifier + regressor) and provides
a single helper function to predict:
- fertilizer type
- fertilizer amount (grams)
- confidence score (if available)

Models are loaded lazily to allow safe importing by Sphinx.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd

# Paths
BASE_DIR = Path(__file__).resolve().parent
CLF_PATH = BASE_DIR / "model_fert_type.joblib"
REG_PATH = BASE_DIR / "model_fert_amount.joblib"

# Lazy-loaded models
_clf = None
_reg = None


def _load_models() -> None:
    """
    Load trained ML models if not already loaded.
    """
    global _clf, _reg

    if _clf is None and CLF_PATH.exists():
        _clf = joblib.load(CLF_PATH)

    if _reg is None and REG_PATH.exists():
        _reg = joblib.load(REG_PATH)


def predict_fertilizer(
    crop: str,
    stage: str,
    soilN: float,
    soilP: float,
    soilK: float,
    ph: float,
    defN: float,
    defP: float,
    defK: float,
) -> Tuple[str, float, float]:
    """
    Predict fertilizer type and amount using supervised ML models.

    Parameters
    ----------
    crop : str
        Crop name.
    stage : str
        Growth stage.
    soilN : float
        Soil nitrogen level.
    soilP : float
        Soil phosphorus level.
    soilK : float
        Soil potassium level.
    ph : float
        Soil pH value.
    defN : float
        Nitrogen deficit (currently unused by ML model).
    defP : float
        Phosphorus deficit (currently unused by ML model).
    defK : float
        Potassium deficit (currently unused by ML model).

    Returns
    -------
    tuple (str, float, float)
        fertilizer_type, fertilizer_amount_g, confidence_percent
    """
    _load_models()

    if _clf is None:
        # Model not available
        return "—", 0.0, 0.0

    X_type = pd.DataFrame([{
        "crop": crop,
        "stage": stage,
        "soilN": float(soilN),
        "soilP": float(soilP),
        "soilK": float(soilK),
        "ph": float(ph),
    }])

    fert_type = str(_clf.predict(X_type)[0])

    # Confidence score
    confidence = 0.0
    if hasattr(_clf, "predict_proba"):
        proba = _clf.predict_proba(X_type)[0]
        confidence = float(proba.max()) * 100.0


    amount = 0.0
    if _reg is not None:
        X_amount = pd.DataFrame([{
            "crop": crop,
            "stage": stage,
            "soilN": float(soilN),
            "soilP": float(soilP),
            "soilK": float(soilK),
            "ph": float(ph),
            "fertilizer_type": fert_type,
        }])
        amount = float(_reg.predict(X_amount)[0])

    if amount < 0:
        amount = 0.0

    return fert_type, round(amount, 2), round(confidence, 1)
