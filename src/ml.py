"""Compare eleven regressors with the same leakage-safe 10-fold pipeline."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.data import FEATURES, ROOT, TARGET, load_dataset


RESULTS = ROOT / "results" / "model_results.csv"
DETAILS = ROOT / "results" / "model_results_detailed.csv"
MODEL_PATH = ROOT / "models" / "best_model.joblib"
K = 10


def make_preprocessor() -> ColumnTransformer:
    numeric = [name for name in FEATURES if name != "node"]
    return ColumnTransformer(
        [
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), ["node"]),
        ]
    )


def models() -> dict:
    return {
        "Ridge": Ridge(),
        "Lasso": Lasso(max_iter=5000),
        "Elastic Net": ElasticNet(max_iter=5000),
        "KNN Regression": KNeighborsRegressor(),
        "Extra Trees Regression": ExtraTreesRegressor(n_estimators=80, n_jobs=1, random_state=42),
        "AdaBoost Regression": AdaBoostRegressor(n_estimators=80, random_state=42),
        "Gradient Boosting Regression": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, n_jobs=1, random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=100, verbosity=-1, n_jobs=1, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=100, verbose=False, thread_count=1, random_seed=42),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, random_state=42),
    }


def train_and_evaluate_models(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_dataset() if df is None else df
    df = df.drop_duplicates().dropna(subset=[TARGET])
    X, y = df[FEATURES], df[TARGET]
    kfold = KFold(n_splits=K, shuffle=True, random_state=42)
    rows = []
    for name, estimator in models().items():
        pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", estimator)])
        scores = cross_validate(
            pipeline, X, y, cv=kfold,
            scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
            n_jobs=1, error_score="raise",
        )
        rmse = -scores["test_rmse"]
        r2 = scores["test_r2"]
        rows.append({
            "Algorithm": name,
            "Number of Features": len(FEATURES),
            "Number of Targets": 1,
            "K-Fold Validation": K,
            "RMSE": float(rmse.mean()),
            "R²": float(r2.mean()),
            "RMSE Std": float(rmse.std()),
            "R² Std": float(r2.std()),
        })
        print(f"{name}: RMSE={rmse.mean():.3f}, R²={r2.mean():.3f}", flush=True)
    detailed = pd.DataFrame(rows).sort_values(["RMSE", "R²"], ascending=[True, False])
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    detailed.to_csv(DETAILS, index=False)
    table = detailed.drop(columns=["RMSE Std", "R² Std"])
    table.to_csv(RESULTS, index=False)
    return table


def create_results_table() -> pd.DataFrame:
    if not RESULTS.exists():
        raise FileNotFoundError("Train models first: python main.py")
    return pd.read_csv(RESULTS)


def select_best_model(table: pd.DataFrame | None = None) -> str:
    table = create_results_table() if table is None else table
    if table.empty:
        raise ValueError("Results table is empty")
    return str(table.sort_values(["RMSE", "R²"], ascending=[True, False]).iloc[0]["Algorithm"])


def save_best_model(name: str, df: pd.DataFrame | None = None) -> Path:
    df = load_dataset() if df is None else df
    pipeline = Pipeline([("preprocessor", make_preprocessor()), ("model", models()[name])])
    pipeline.fit(df[FEATURES], df[TARGET])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH, compress=3)
    return MODEL_PATH


def save_plots(df: pd.DataFrame, table: pd.DataFrame) -> None:
    plot_dir = RESULTS.parent / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[TARGET], bins=35)
    ax.set(title="Distribution of Nodal Electricity Demand", xlabel="Demand (source units)", ylabel="Rows")
    fig.tight_layout()
    fig.savefig(plot_dir / "target_distribution.png", dpi=140)
    plt.close(fig)
    for metric, filename in [("RMSE", "rmse_comparison.png"), ("R²", "r2_comparison.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(table["Algorithm"], table[metric])
        ax.invert_yaxis()
        ax.set(title=f"{metric} by Algorithm", xlabel=metric)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=140)
        plt.close(fig)
