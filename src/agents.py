"""Deep Agents orchestration around the tested Python data and ML functions."""

import json
import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from src.data import FEATURES, TARGET, load_dataset, prepare_dataset
from src.ml import RESULTS, create_results_table, save_best_model, select_best_model, train_and_evaluate_models
from src.prediction import predict_energy_consumption


@tool
def load_dataset_tool() -> str:
    """Load the prepared Kazakhstan nodal electricity demand dataset and report its size."""
    df = load_dataset()
    return json.dumps({"rows": len(df), "columns": df.columns.tolist(), "target": TARGET})


@tool
def inspect_dataset_tool() -> str:
    """Report dataset columns, missing values, duplicates, and demand distribution."""
    df = load_dataset()
    return json.dumps({
        "rows": len(df), "features": FEATURES, "target": TARGET,
        "missing_values": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "target_description": df[TARGET].describe().to_dict(),
    })


@tool
def prepare_dataset_tool() -> str:
    """Rebuild data/energy.csv from the published original Excel workbook."""
    df = prepare_dataset()
    return json.dumps({"rows": len(df), "features": FEATURES, "target": TARGET})


@tool
def train_and_evaluate_models_tool() -> str:
    """Train all 11 regressors with 10-fold CV and save their actual RMSE/R2 results."""
    table = train_and_evaluate_models()
    return table.to_json(orient="records")


@tool
def create_results_table_tool() -> str:
    """Read the calculated Table 1 from results/model_results.csv."""
    return create_results_table().to_json(orient="records")


@tool
def select_best_model_tool() -> str:
    """Select the smallest-RMSE model from saved CV results and fit/save its full pipeline."""
    name = select_best_model()
    path = save_best_model(name)
    return json.dumps({"best_model": name, "pipeline": str(path), "results": str(RESULTS)})


@tool
def predict_energy_consumption_tool(week: int, hour_in_week: int, node: str) -> str:
    """Predict source-scale nodal electricity demand with the saved ML pipeline."""
    value = predict_energy_consumption({"week": week, "hour_in_week": hour_in_week, "node": node})
    return json.dumps({"week": week, "hour_in_week": hour_in_week, "node": node, "prediction": value})


SUBAGENTS = [
    {
        "name": "data_agent",
        "description": "Inspect, load, and prepare the Kazakhstan electricity demand dataset.",
        "system_prompt": "Use your Python tools for all data facts. Explain that the published nodal series includes modeled allocation of observed demand.",
        "tools": [load_dataset_tool, inspect_dataset_tool, prepare_dataset_tool],
    },
    {
        "name": "training_agent",
        "description": "Run all eleven regression algorithms with shared 10-fold validation.",
        "system_prompt": "Call the training tool. Report only metrics returned by it. Training can take several minutes.",
        "tools": [train_and_evaluate_models_tool],
    },
    {
        "name": "evaluation_agent",
        "description": "Read Table 1, compare RMSE and R2, and save the winning pipeline.",
        "system_prompt": "Call tools to obtain actual results. Select minimum RMSE, breaking a tie by larger R2. Never invent scores.",
        "tools": [create_results_table_tool, select_best_model_tool],
    },
    {
        "name": "prediction_agent",
        "description": "Validate a week, hour, and network node; predict demand with the saved ML pipeline.",
        "system_prompt": "Ask for missing week (1 or 2), hour_in_week (1..168), or node. Call the prediction tool; never calculate the prediction yourself.",
        "tools": [predict_energy_consumption_tool],
    },
]


def create_agent():
    load_dotenv()
    provider = os.getenv("MODEL_PROVIDER", "openai").strip()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
    key_name = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google_genai": "GOOGLE_API_KEY"}.get(provider)
    if key_name and not os.getenv(key_name):
        raise RuntimeError(f"Set {key_name} in .env before starting Deep Agents")
    if not provider or not model_name:
        raise RuntimeError("Set MODEL_PROVIDER and MODEL_NAME in .env")
    model = init_chat_model(model_name, model_provider=provider)
    return create_deep_agent(
        model=model,
        subagents=SUBAGENTS,
        system_prompt=(
            "You coordinate four specialized agents. Delegate dataset requests to data_agent, "
            "training to training_agent, results to evaluation_agent, and predictions to "
            "prediction_agent. The Python tools are the only source of numerical results. "
            "The published two-week nodal series is a processed research dataset, not live measurements."
        ),
    )


def run_chat() -> None:
    agent = create_agent()
    print("Deep Agents ready. Ask about data, training, evaluation, or prediction. Type exit to stop.")
    while True:
        request = input("You> ").strip()
        if request.lower() in {"exit", "quit"}:
            break
        if not request:
            continue
        response = agent.invoke({"messages": [{"role": "user", "content": request}]})
        print("Agent>", response["messages"][-1].content)


if __name__ == "__main__":
    run_chat()
