import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"
LAUNCHER_FILE = ROOT / "run_three_station_yearly_protocol.py"
PLOT_FILE = ROOT / "scripts" / "plot_ft_sweep_curves.py"


class FedTLFineTuneSweepAstTest(unittest.TestCase):
    def test_training_declares_fedavg_and_ft_sweep_surface(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        for token in [
            "FED_PRETRAIN_ALGO",
            "FEDAVG_LOCAL_EPOCHS",
            "FEDAVG_CLIENT_WEIGHTING",
            "run_fedavg_pretraining_round",
            "aggregate_client_states",
            'FEDAVG_CLIENT_WEIGHTING == "sample"',
            'if FED_PRETRAIN_ALGO == "fedavg":',
            "ENABLE_FT_SWEEP",
            "FT_SWEEP_EPOCHS",
            "FT_SWEEP_OUTPUT_PATH",
            "parse_ft_sweep_epochs",
            "build_ft_sweep_eval_fn",
            "make_ft_sweep_checkpoint_path",
            "export_ft_sweep_records",
            "Original-Local-FT",
            "FedTL-FT",
            "model_fore_station{station_id}_extreme{class_idx}_{model_tag}_ft_epoch{epoch_value}.pth",
        ]:
            self.assertIn(token, text)

    def test_launcher_exposes_fedavg_and_ft_sweep_env(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "FED_PRETRAIN_ALGO",
            "FEDAVG_LOCAL_EPOCHS",
            "FEDAVG_CLIENT_WEIGHTING",
            "FEDAVG_LR",
            "ENABLE_FT_SWEEP",
            "FT_SWEEP_EPOCHS",
            "FT_SWEEP_OUTPUT_PATH",
            "FT_SWEEP_SAVE_CHECKPOINTS",
            "FT_SWEEP_EVAL_TEST",
        ]:
            self.assertIn(token, text)

    def test_plot_script_reads_sweep_csv(self):
        text = PLOT_FILE.read_text(encoding="utf-8")

        for token in [
            "--csv",
            "--label",
            "--output",
            "SupportAll_MSE",
            "Val_MSE",
            "Test_MSE",
            "Test_nMAE_%",
            "weighted_average",
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
