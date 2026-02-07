"""
Dataset generator for supervised fertilizer models.

This module creates a synthetic dataset using the rule-based expert system
(rules_engine.expert_recommendation) to produce labels for:
- fertilizer_type
- fertilizer_amount_g

The output CSV is used for training supervised models.
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from data import IDEAL, CROPS
from rules_engine import expert_recommendation


def generate_rows(per_crop: int = 400, seed: int = 42) -> list[dict[str, Any]]:
    """
    Generate synthetic training rows for each crop.

    Parameters
    ----------
    per_crop : int, optional
        Number of rows to generate per crop, by default 400.
    seed : int, optional
        Random seed for reproducibility, by default 42.

    Returns
    -------
    list[dict[str, Any]]
        List of row dictionaries containing features and labels.
    """
    random.seed(seed)
    rows: list[dict[str, Any]] = []

    for crop in CROPS:
        stages = list(IDEAL[crop].keys())

        for _ in range(per_crop):
            stage = random.choice(stages)
            ideal = IDEAL[crop][stage]

            soilN = max(0.0, random.uniform(0.0, ideal["N"] * 1.2))
            soilP = max(0.0, random.uniform(0.0, ideal["P"] * 1.2))
            soilK = max(0.0, random.uniform(0.0, ideal["K"] * 1.2))

            ph = random.uniform(5.0, 8.0)

            rec = expert_recommendation(crop, stage, soilN, soilP, soilK)

            rows.append({
                "crop": crop,
                "stage": stage,
                "soilN": round(soilN, 2),
                "soilP": round(soilP, 2),
                "soilK": round(soilK, 2),
                "ph": round(ph, 2),

                "defN": rec["defN"],
                "defP": rec["defP"],
                "defK": rec["defK"],

                "fertilizer_type": rec["fertilizer_type"],
                "fertilizer_amount_g": rec["fertilizer_amount_g"],
            })

    return rows


def main() -> None:
    """
    Generate the dataset and write it to multi_supervised.csv.
    """
    rows = generate_rows(per_crop=400)
    df = pd.DataFrame(rows)
    df.to_csv("multi_supervised.csv", index=False, encoding="utf-8-sig")
    print("Saved dataset: multi_supervised.csv rows=", len(df))


if __name__ == "__main__":
    main()
