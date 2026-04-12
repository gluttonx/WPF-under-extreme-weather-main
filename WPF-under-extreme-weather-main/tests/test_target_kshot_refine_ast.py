import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"
LAUNCHER_FILE = ROOT / "run_three_station_yearly_protocol.py"


class TargetKShotRefineAstTest(unittest.TestCase):
    def test_training_script_exposes_target_kshot_config(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("EXTREME_TARGET_ADAPT_MAX_WINDOWS", text)
        self.assertIn("apply_target_adapt_kshot_limit", text)
        self.assertIn('"extreme_target_adapt_max_windows": EXTREME_TARGET_ADAPT_MAX_WINDOWS', text)

    def test_target_kshot_limit_only_applies_to_target_split(self):
        tree = ast.parse(TRAIN_FILE.read_text(encoding="utf-8"))
        calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            if not isinstance(func, ast.Name) or func.id != "apply_target_adapt_kshot_limit":
                continue
            targets = [target.id for target in node.targets if isinstance(target, ast.Name)]
            calls.extend(targets)

        self.assertIn("target_split_payload", calls)
        self.assertNotIn("source_split_payload", calls)

    def test_launcher_has_refine_only_kshot_presets(self):
        text = LAUNCHER_FILE.read_text(encoding="utf-8")

        for token in [
            "refine-k1",
            "refine-k2",
            '"SKIP_LOCAL_PRETRAIN": "1"',
            '"SKIP_LOCAL_META": "1"',
            '"EXTREME_TARGET_ADAPT_MAX_WINDOWS": "1"',
            '"EXTREME_TARGET_ADAPT_MAX_WINDOWS": "2"',
        ]:
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
