import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class RegimePriorCouplingAstTest(unittest.TestCase):
    def test_training_script_contains_regime_aware_fed_pretrain_hooks(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("def env_float(", text)
        self.assertIn("FED_PRETRAIN_REGIME_ALPHA", text)
        self.assertIn("FED_PRETRAIN_AGGREGATION_GAMMA", text)
        self.assertIn("def compute_regime_sample_weights(", text)
        self.assertIn("def weighted_mse_loss(", text)
        self.assertIn("regime_factor", text)
        self.assertIn("aggregation_weight", text)

    def test_training_script_contains_prior_preserving_local_meta_hooks(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("PROPOSED_META_SHARED_ANCHOR_BETA", text)
        self.assertIn("PROPOSED_META_SHARED_LR_SCALE", text)
        self.assertIn("def build_meta_optimizer(", text)
        self.assertIn("def compute_shared_anchor_loss(", text)
        self.assertIn("shared_anchor_beta", text)
        self.assertIn("shared_lr_scale", text)
        self.assertIn("loss_en_q = loss_en_q + shared_anchor_beta * anchor_loss_q", text)

    def test_proposed_and_baselines_use_different_meta_regularization(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("shared_anchor_beta=PROPOSED_META_SHARED_ANCHOR_BETA", text)
        self.assertIn("shared_lr_scale=PROPOSED_META_SHARED_LR_SCALE", text)
        self.assertIn("shared_anchor_beta=0.0", text)
        self.assertIn("shared_lr_scale=1.0", text)


if __name__ == "__main__":
    unittest.main()
