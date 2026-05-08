import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVAL_FILE = ROOT / "generate_multi_station_results.py"


class TwoYear2024WeightedOverallAstTest(unittest.TestCase):
    def test_eval_exports_sample_weighted_overall_summary(self):
        text = EVAL_FILE.read_text(encoding="utf-8")

        for token in [
            "def weighted_metric_average(group_df, column_name):",
            "Overall_SampleWeighted",
            "weighted_metric_average(g, 'nMAE_%')",
            "weighted_metric_average(g, 'nRMSE_%')",
            "weighted_metric_average(g, 'WD_%')",
            "weighted_metric_average(g, 'R_p<0.05_%')",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
