"""Small regression checks for the assignment artifacts and Python tools."""

import json
import math
import unittest

import joblib

from src.agents import SUBAGENTS, predict_energy_consumption_tool, verify_dataset_tool, verify_results_tool
from src.data import FEATURES, TARGET, load_dataset
from src.ml import MODEL_PATH, RESULTS, create_results_table, models, select_best_model
from src.prediction import predict_energy_consumption
from src.data import ROOT


class ProjectScenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_dataset()
        cls.results = create_results_table()

    def test_published_table_has_enough_rows(self):
        self.assertEqual(len(self.data), 11088)

    def test_feature_and_target_schema(self):
        self.assertEqual(list(self.data.columns), FEATURES + [TARGET])

    def test_data_quality(self):
        self.assertEqual(int(self.data.isna().sum().sum()), 0)
        self.assertEqual(int(self.data.duplicated().sum()), 0)

    def test_all_required_algorithms(self):
        self.assertEqual(len(models()), 11)
        self.assertEqual(set(self.results["Algorithm"]), set(models()))

    def test_shared_ten_fold_results(self):
        self.assertTrue((self.results["K-Fold Validation"] == 10).all())
        self.assertTrue(self.results[["RMSE", "R²"]].notna().all().all())

    def test_best_model_is_selected_from_scores(self):
        best = select_best_model(self.results)
        self.assertEqual(best, self.results.sort_values(["RMSE", "R²"], ascending=[True, False]).iloc[0]["Algorithm"])

    def test_saved_pipeline_has_preprocessing_and_model(self):
        self.assertTrue(MODEL_PATH.exists() and RESULTS.exists())
        self.assertEqual(set(joblib.load(MODEL_PATH).named_steps), {"preprocessor", "model"})

    def test_prediction_is_numeric(self):
        self.assertTrue(math.isfinite(predict_energy_consumption({"week": 1, "hour_in_week": 1, "node": "N0000"})))

    def test_invalid_week_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_energy_consumption({"week": 3, "hour_in_week": 1, "node": "N0000"})

    def test_invalid_hour_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_energy_consumption({"week": 1, "hour_in_week": 169, "node": "N0000"})

    def test_unknown_node_is_rejected(self):
        with self.assertRaises(ValueError):
            predict_energy_consumption({"week": 1, "hour_in_week": 1, "node": "UNKNOWN"})

    def test_five_specialists_have_at_most_five_tools(self):
        self.assertEqual(len(SUBAGENTS), 5)
        self.assertEqual(len({agent["name"] for agent in SUBAGENTS}), 5)
        self.assertTrue(all(1 <= len(agent["tools"]) <= 5 for agent in SUBAGENTS))

    def test_independent_verification_tools(self):
        self.assertIn('"passed": true', verify_dataset_tool.invoke({}))
        self.assertIn('"passed": true', verify_results_tool.invoke({}))

    def test_prediction_tool_returns_clear_error(self):
        result = json.loads(predict_energy_consumption_tool.invoke({
            "week": 3, "hour_in_week": 1, "node": "N0000"
        }))
        self.assertEqual(result["status"], "error")
        self.assertIn("outside", result["message"])

    def test_sample_run_proves_bounded_agent_share(self):
        sample = ROOT / "docs" / "sample_run.jsonl"
        summary = json.loads((ROOT / "docs" / "sample_run_summary.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for line in sample.read_text(encoding="utf-8").splitlines()]
        counts = {}
        for event in events:
            if event["event"] in {"llm_call", "tool_call"}:
                counts[event["agent"]] = counts.get(event["agent"], 0) + 1
        self.assertEqual(counts, summary["counts"])
        self.assertEqual(set(counts), {"main_agent"} | {agent["name"] for agent in SUBAGENTS})
        self.assertLessEqual(max(counts.values()) / sum(counts.values()), 0.4)


if __name__ == "__main__":
    unittest.main()
