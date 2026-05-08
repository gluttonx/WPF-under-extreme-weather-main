import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAGNOSTIC_FILE = ROOT / "scripts" / "run_high_wind_pretrain_diagnostic.py"


class HighWindPretrainDiagnosticAstTest(unittest.TestCase):
    def test_diagnostic_declares_expected_variants(self):
        text = DIAGNOSTIC_FILE.read_text(encoding="utf-8")

        for token in [
            "constant_0p9",
            "train_mean",
            "train_median",
            "long_sequence",
            "window6",
            "window6_highwind_weighted",
            "VariantConfig",
        ]:
            self.assertIn(token, text)

    def test_diagnostic_uses_high_wind_protocol_payloads(self):
        text = DIAGNOSTIC_FILE.read_text(encoding="utf-8")

        for token in [
            "protocol_data/high_wind_spring_noft_four_client",
            "protocol_metadata.json",
            "nwp_1h",
            "p_1h",
            "nwp_test_extre_class1_",
            "p_test_extre_class1",
            "train_scales",
            "normal_nwp / train_scales",
            "test_nwp_raw / train_scales",
        ]:
            self.assertIn(token, text)

    def test_diagnostic_has_weighted_window_loss_and_no_formal_meta_training(self):
        text = DIAGNOSTIC_FILE.read_text(encoding="utf-8")

        for token in [
            "def weighted_mse",
            "high_wind_threshold",
            "high_wind_weight",
            "make_window_tensor",
            "raw_wind >= high_wind_threshold",
            "ModelFore(mode=\"pre\")",
        ]:
            self.assertIn(token, text)

        self.assertNotIn("run_fed_normal_meta_training", text)
        self.assertNotIn("test_task_support/Few-shot", text)

    def test_diagnostic_writes_reproducible_csv_outputs(self):
        text = DIAGNOSTIC_FILE.read_text(encoding="utf-8")

        for token in [
            "diagnostic_results.csv",
            "diagnostic_overall.csv",
            "diagnostic_predictions.csv",
            "set_reproducible_seed",
            "--smoke",
            "--save-models",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
