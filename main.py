"""Run the complete assignment without requiring an LLM key."""

import argparse
import sys

from src.data import inspect_dataset, prepare_dataset
from src.ml import save_best_model, save_plots, select_best_model, train_and_evaluate_models
from src.prediction import predict_energy_consumption


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", action="store_true", help="Start interactive Deep Agents chat after ML training")
    args = parser.parse_args()

    print("Preparing published Kazakhstan electricity demand data", flush=True)
    df = prepare_dataset()
    inspect_dataset(df)
    print("\nTraining 11 models with shared 10-fold cross-validation", flush=True)
    table = train_and_evaluate_models(df)
    print("\nTABLE 1\n", table.to_string(index=False))
    best = select_best_model(table)
    path = save_best_model(best, df)
    save_plots(df, table)
    example = {"week": 1, "hour_in_week": 1, "node": "N0000"}
    prediction = predict_energy_consumption(example)
    print(f"\nBest model: {best}\nSaved: {path}")
    print(f"Example prediction for {example}: {prediction:.3f} source demand units")

    if args.agent:
        from src.agents import run_chat
        run_chat()


if __name__ == "__main__":
    main()
