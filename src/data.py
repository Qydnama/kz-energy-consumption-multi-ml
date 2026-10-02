"""Load the published Kazakhstan power system workbook and inspect the ML table."""

from pathlib import Path
import io

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_kazakhstan_power_system.xlsx"
DATASET = ROOT / "data" / "energy.csv"
FEATURES = ["week", "hour_in_week", "hour_of_day", "day_in_week", "node"]
TARGET = "demand"


def prepare_dataset() -> pd.DataFrame:
    """Reshape the original DEMAND sheet; no target-derived features are added."""
    if not SOURCE.exists():
        raise FileNotFoundError(f"Published workbook is missing: {SOURCE}")
    wide = pd.read_excel(SOURCE, sheet_name="DEMAND", header=1)
    node_columns = [column for column in wide if str(column).startswith("N")]
    if len(node_columns) != 33:
        raise ValueError(f"Expected 33 demand nodes, found {len(node_columns)}")
    table = wide.melt(
        id_vars=["week", "hour"],
        value_vars=node_columns,
        var_name="node",
        value_name=TARGET,
    ).rename(columns={"hour": "hour_in_week"})
    table["hour_of_day"] = ((table["hour_in_week"] - 1) % 24 + 1).astype(int)
    table["day_in_week"] = ((table["hour_in_week"] - 1) // 24 + 1).astype(int)
    table[TARGET] = pd.to_numeric(table[TARGET], errors="coerce")
    table = table.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    table = table[FEATURES + [TARGET]]
    if len(table) < 10 or table[TARGET].nunique() < 2:
        raise ValueError("Dataset does not support 10-fold regression")
    DATASET.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(DATASET, index=False)
    return table


def load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        return prepare_dataset()
    return pd.read_csv(DATASET)


def inspect_dataset(df: pd.DataFrame | None = None) -> dict:
    df = load_dataset() if df is None else df
    buffer = io.StringIO()
    df.info(buf=buffer)
    print("\nHEAD\n", df.head().to_string(index=False))
    print("\nSHAPE\n", df.shape)
    print("\nINFO\n", buffer.getvalue())
    print("\nDESCRIBE\n", df.describe(include="all").to_string())
    print("\nMISSING VALUES\n", df.isnull().sum().to_string())
    print("\nDUPLICATES\n", df.duplicated().sum())
    summary = {
        "rows": len(df),
        "features": FEATURES,
        "target": TARGET,
        "numerical_features": [name for name in FEATURES if name != "node"],
        "categorical_features": ["node"],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }
    print("\nSUMMARY\n", summary)
    return summary
