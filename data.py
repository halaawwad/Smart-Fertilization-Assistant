"""
Static reference data for the fertilizer recommendation system.

This module contains:
- IDEAL: ideal N/P/K targets by crop and growth stage
- CROPS: list of supported crops
- STAGE_LABELS_AR: Arabic labels for stages (for UI display)
- UNIT: unit label used across the app
- PH_RANGES: optimal/suboptimal pH ranges per crop
"""

from __future__ import annotations

# Ideal NPK targets (grams per dunum per day) by crop and stage
IDEAL = {
    "cucumber": {
        "transplant_0_14_days": {"N": 150, "P": 150, "K": 150},
        "14_35_days": {"N": 250, "P": 150, "K": 400},
        "35_days_to_end": {"N": 425, "P": 250, "K": 670},
    },
    "tomato": {
        "transplant_to_flowering": {"N": 150, "P": 150, "K": 150},
        "flowering_cluster_1_3": {"N": 350, "P": 200, "K": 540},
        "flowering_cluster_4_6": {"N": 425, "P": 250, "K": 670},
        "fruit_development": {"N": 560, "P": 330, "K": 900},
    },
}

# Supported crops (keys of IDEAL)
CROPS = list(IDEAL.keys())

# Arabic labels for UI display
STAGE_LABELS_AR = {
    "cucumber": {
        "transplant_0_14_days": "تشتيل – 14 يوم (خيار)",
        "14_35_days": "14 – 35 يوم (خيار)",
        "35_days_to_end": "35 يوم – نهاية المحصول (خيار)",
    },
    "tomato": {
        "transplant_to_flowering": "تشتيل – بداية الإزهار (بندورة)",
        "flowering_cluster_1_3": "إزهار – حتى الفوج 1–3 (بندورة)",
        "flowering_cluster_4_6": "إزهار – حتى الفوج 4–6 (بندورة)",
        "fruit_development": "تطور الثمار والقطف (بندورة)",
    },
}

# Unit label used across the application
UNIT = "g/dunum/day"

# pH ranges per crop: optimal and acceptable (suboptimal) ranges
PH_RANGES = {
    "cucumber": {"optimal": (6.0, 6.8), "suboptimal": (5.5, 7.5)},
    "tomato": {"optimal": (6.0, 6.8), "suboptimal": (5.5, 7.5)},
}

# Optional: explicit export list (helps documentation/readability)
__all__ = ["IDEAL", "CROPS", "STAGE_LABELS_AR", "UNIT", "PH_RANGES"]
