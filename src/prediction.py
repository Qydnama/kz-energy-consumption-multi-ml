"""Load the saved preprocessing/model pipeline for one nodal demand prediction."""

import joblib
import pandas as pd

from src.data import FEATURES, load_dataset
from src.ml import MODEL_PATH


def predict_energy_consumption(input_data: dict) -> float:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Train and save a model first: {MODEL_PATH}")
    try:
        week = int(input_data["week"])
        hour = int(input_data["hour_in_week"])
        node = str(input_data["node"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Provide week (1 or 2), hour_in_week (1..168), and node") from exc
    known_nodes = set(load_dataset()["node"])
    if week not in (1, 2) or not 1 <= hour <= 168 or node not in known_nodes:
        raise ValueError("Input is outside the two source weeks or known network nodes")
    row = pd.DataFrame([{
        "week": week,
        "hour_in_week": hour,
        "hour_of_day": (hour - 1) % 24 + 1,
        "day_in_week": (hour - 1) // 24 + 1,
        "node": node,
    }], columns=FEATURES)
    model = joblib.load(MODEL_PATH)
    return float(model.predict(row)[0])
