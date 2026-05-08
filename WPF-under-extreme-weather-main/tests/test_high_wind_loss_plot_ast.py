import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLOT_FILE = ROOT / "scripts" / "plot_high_wind_loss_curves.py"


class HighWindLossPlotAstTest(unittest.TestCase):
    def test_plot_script_declares_epoch_based_high_wind_curves(self):
        text = PLOT_FILE.read_text(encoding="utf-8")
        for token in [
            "High-Wind Spring NoFT Training Loss by Epoch",
            "loss_mse_pre_station",
            "loss_mse_target_aware_pre_station",
            "loss_mse_train_task_query_local_meta_station",
            "loss_mse_train_task_query_target_aware_meta_station",
            "loss_mse_target_aware_selective_fed_meta_proxy_station",
            "loss_mse_target_aware_selective_fed_meta_self_proxy_station",
            "loss_mse_train_task_query_target_aware_selective_fed_meta_station",
            "Legacy candidate query MSE",
            "--pretrain-history-artifact-dir",
            "pretrain_only=True",
            "PRETRAIN_STAGES",
            "drop_duplicates",
            "ax.set_xlabel(\"Epoch\")",
            "ax.set_ylabel(\"MSE loss\")",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
