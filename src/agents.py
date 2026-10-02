"""Deep Agents orchestration around the tested Python data and ML functions."""

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from uuid import uuid4

import joblib
from deepagents import (
    GeneralPurposeSubagentProfile,
    HarnessProfileConfig,
    create_deep_agent,
    register_harness_profile,
)
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from src.data import FEATURES, ROOT, TARGET, load_dataset, prepare_dataset
from src.ml import MODEL_PATH, RESULTS, create_results_table, save_best_model, select_best_model
from src.prediction import predict_energy_consumption


LOG_DIR = ROOT / "results" / "agent_runs"
MAX_STEPS = 50
REQUEST_TIMEOUT_SECONDS = 300
TRAINING_TIMEOUT_SECONDS = 180
MODEL_TIMEOUT_SECONDS = 45


@tool
def load_dataset_tool() -> str:
    """Load the prepared Kazakhstan nodal electricity demand dataset and report its size."""
    try:
        df = load_dataset()
        return json.dumps({"status": "ok", "rows": len(df), "columns": df.columns.tolist(), "target": TARGET})
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def inspect_dataset_tool() -> str:
    """Report dataset columns, missing values, duplicates, and demand distribution."""
    try:
        df = load_dataset()
        return json.dumps({
            "status": "ok", "rows": len(df), "features": FEATURES, "target": TARGET,
            "missing_values": df.isnull().sum().to_dict(),
            "duplicates": int(df.duplicated().sum()),
            "target_description": df[TARGET].describe().to_dict(),
        })
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def prepare_dataset_tool() -> str:
    """Rebuild data/energy.csv from the published original Excel workbook."""
    try:
        df = prepare_dataset()
        return json.dumps({"status": "ok", "rows": len(df), "features": FEATURES, "target": TARGET})
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def train_and_evaluate_models_tool() -> str:
    """Train all 11 regressors with 10-fold CV and save their actual RMSE/R2 results."""
    try:
        subprocess.run(
            [sys.executable, "-m", "src.ml"], cwd=ROOT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, check=True, timeout=TRAINING_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return json.dumps({"status": "error", "message": "Training exceeded the 180-second limit"})
    except subprocess.CalledProcessError:
        return json.dumps({"status": "error", "message": "Training failed; run python main.py to inspect the ML error"})
    table = create_results_table()
    return json.dumps({
        "status": "ok",
        "results_file": str(RESULTS.relative_to(ROOT)),
        "results": json.loads(table.to_json(orient="records")),
    })


@tool
def create_results_table_tool() -> str:
    """Read the calculated Table 1 from results/model_results.csv."""
    try:
        table = create_results_table()
        return json.dumps({"status": "ok", "results": json.loads(table.to_json(orient="records"))})
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def select_best_model_tool() -> str:
    """Select the smallest-RMSE model from saved CV results and fit/save its full pipeline."""
    try:
        name = select_best_model()
        path = save_best_model(name)
        return json.dumps({"status": "ok", "best_model": name, "pipeline": str(path), "results": str(RESULTS)})
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def predict_energy_consumption_tool(
    week: int | None = None, hour_in_week: int | None = None, node: str | None = None
) -> str:
    """Predict source-scale nodal electricity demand with the saved ML pipeline."""
    try:
        value = predict_energy_consumption({"week": week, "hour_in_week": hour_in_week, "node": node})
        return json.dumps({"status": "ok", "week": week, "hour_in_week": hour_in_week, "node": node, "prediction": value})
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"status": "error", "message": str(exc)})


@tool
def verify_dataset_tool() -> str:
    """Check that the prepared dataset has the expected schema, rows, and no missing target."""
    try:
        df = load_dataset()
        passed = len(df) == 11088 and list(df.columns) == FEATURES + [TARGET] and df[TARGET].notna().all()
        return json.dumps({"status": "ok" if passed else "error", "passed": bool(passed), "rows": len(df), "columns": df.columns.tolist()})
    except (FileNotFoundError, ValueError) as exc:
        return json.dumps({"status": "error", "passed": False, "message": str(exc)})


@tool
def verify_results_tool() -> str:
    """Check all eleven CV results, the saved pipeline, and one finite ML prediction."""
    try:
        table = create_results_table()
        pipeline = joblib.load(MODEL_PATH)
        prediction = predict_energy_consumption({"week": 1, "hour_in_week": 1, "node": "N0000"})
        passed = (
            len(table) == 11
            and (table["K-Fold Validation"] == 10).all()
            and table["RMSE"].notna().all()
            and table["R²"].notna().all()
            and set(pipeline.named_steps) == {"preprocessor", "model"}
            and math.isfinite(prediction)
        )
        return json.dumps({
            "status": "ok" if passed else "error", "passed": bool(passed),
            "algorithms": len(table), "best_model": select_best_model(table), "prediction": prediction,
        })
    except (FileNotFoundError, ValueError, KeyError, AttributeError) as exc:
        return json.dumps({"status": "error", "passed": False, "message": str(exc)})


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
        "system_prompt": "Call the training tool. Report only metrics and the exact results_file returned by it. Training can take several minutes.",
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
    {
        "name": "verification_agent",
        "description": "Independently check data, CV results, saved model, and prediction artifacts.",
        "system_prompt": "Call both verification tools. Return their actual pass/fail results as JSON. Do not change any artifact.",
        "tools": [verify_dataset_tool, verify_results_tool],
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
    register_harness_profile(
        f"{provider}:{model_name}",
        HarnessProfileConfig(
            excluded_tools=frozenset({"ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute"}),
            general_purpose_subagent=GeneralPurposeSubagentProfile(enabled=False),
        ),
    )
    model = init_chat_model(
        model_name, model_provider=provider,
        timeout=MODEL_TIMEOUT_SECONDS, max_retries=2,
    )
    return create_deep_agent(
        model=model,
        checkpointer=InMemorySaver(),
        subagents=SUBAGENTS,
        system_prompt=(
            "You are an orchestrator. Only plan and delegate; do no data, ML, evaluation, "
            "prediction, or verification work yourself. Delegate dataset requests to data_agent, "
            "training to training_agent, results to evaluation_agent, and predictions to "
            "prediction_agent. Delegate artifact checks to verification_agent. "
            "When asked for a full workflow, call all five specialists in order, one at a time. "
            "Put task, inputs, and expected_output in a JSON object inside each task description. "
            "The Python tools are the only source of numerical results. "
            "Only report file paths that a tool explicitly returns; never invent artifact names. "
            "The published two-week nodal series is a processed research dataset, not live measurements."
        ),
    )


def run_request(agent, request: str, thread_id: str) -> dict:
    """Run one request, retaining chat state and logging every LLM/tool call."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_id = uuid4().hex[:12]
    log_path = LOG_DIR / f"{run_id}.jsonl"
    counts: dict[str, int] = {}
    pending_roles: list[str] = []
    namespace_roles: dict[tuple, str] = {}
    final_answer = None
    started = time.monotonic()

    with log_path.open("w", encoding="utf-8") as log:
        def record(agent_name: str, event: str, **fields) -> None:
            entry = {
                "time": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "agent": agent_name,
                "event": event,
                **fields,
            }
            log.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log.flush()
            if event in {"llm_call", "tool_call"}:
                counts[agent_name] = counts.get(agent_name, 0) + 1

        record("system", "start", thread_id=thread_id)
        try:
            stream = agent.stream(
                {"messages": [{"role": "user", "content": request}]},
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": MAX_STEPS},
                stream_mode="updates", subgraphs=True,
            )
            for namespace, chunk in stream:
                if time.monotonic() - started > REQUEST_TIMEOUT_SECONDS:
                    stream.close()
                    raise TimeoutError("Agent request exceeded the 300-second limit")
                if namespace:
                    if namespace not in namespace_roles:
                        namespace_roles[namespace] = pending_roles.pop(0) if pending_roles else "unmapped_subagent"
                    role = namespace_roles[namespace]
                else:
                    role = "main_agent"
                for node, value in chunk.items():
                    messages = value.get("messages", []) if isinstance(value, dict) else []
                    for message in messages:
                        if node == "model":
                            record(role, "llm_call", status="completed")
                            calls = getattr(message, "tool_calls", None) or []
                            for call in calls:
                                if role == "main_agent" and call["name"] == "task":
                                    pending_roles.append(call.get("args", {}).get("subagent_type", "unknown"))
                                record(role, "tool_call", tool=call["name"], status="requested")
                            if role == "main_agent" and not calls:
                                final_answer = str(message.content)
                        elif node == "tools":
                            status = getattr(message, "status", "success")
                            try:
                                payload = json.loads(str(message.content))
                                if isinstance(payload, dict) and payload.get("status") == "error":
                                    status = "error"
                            except (TypeError, ValueError):
                                pass
                            record(role, "tool_result", tool=getattr(message, "name", "unknown"),
                                   status=status)
            if final_answer is None:
                raise RuntimeError("Main agent returned no final answer")
            record("system", "finish", elapsed_seconds=round(time.monotonic() - started, 2))
        except Exception as exc:
            record("system", "error", error_type=type(exc).__name__)
            reason = {
                "TimeoutError": "Agent request timed out",
                "GraphRecursionError": "Agent exceeded the step limit",
            }.get(type(exc).__name__, f"Agent request failed ({type(exc).__name__})")
            raise RuntimeError(f"{reason}; see {log_path}") from exc

    total = sum(counts.values())
    shares = {name: round(count / total, 3) for name, count in counts.items()} if total else {}
    summary = {
        "run_id": run_id,
        "log_file": str(log_path.relative_to(ROOT)),
        "counts": counts,
        "shares": shares,
        "balanced": bool(counts) and max(counts.values()) / total <= 0.4,
        "answer": final_answer,
    }
    saved_summary = {key: value for key, value in summary.items() if key != "answer"}
    log_path.with_suffix(".summary.json").write_text(
        json.dumps(saved_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


DEMO_REQUEST = (
    "Run the complete Kazakhstan electricity demand workflow. In order, delegate to "
    "data_agent to inspect the dataset, training_agent to train all 11 models with 10-fold CV, "
    "evaluation_agent to read Table 1 and save the best model, prediction_agent to predict "
    "week 1 hour_in_week 1 node N0000, and verification_agent to check the dataset and artifacts. "
    "Call all five specialists one at a time using their tools. Report the best model, RMSE, "
    "prediction, verification pass/fail, and exact results path."
)


def run_chat() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run the five-agent scenario once")
    args, _ = parser.parse_known_args()
    agent = create_agent()
    thread_id = uuid4().hex
    if args.demo:
        summary = run_request(agent, DEMO_REQUEST, thread_id)
        print(summary["answer"])
        print("Results:", RESULTS.relative_to(ROOT))
        print("Run log:", summary["log_file"])
        print("Call shares:", summary["shares"], "balanced:", summary["balanced"])
        return
    print("Deep Agents ready. Ask about data, training, evaluation, or prediction. Type exit to stop.")
    while True:
        request = input("You> ").strip()
        if request.lower() in {"exit", "quit"}:
            break
        if request:
            try:
                summary = run_request(agent, request, thread_id)
                print("Agent>", summary["answer"])
                print("Run log:", summary["log_file"])
            except RuntimeError as exc:
                print(exc)


if __name__ == "__main__":
    try:
        run_chat()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
