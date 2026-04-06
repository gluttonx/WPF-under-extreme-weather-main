import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENTS_FILE = ROOT / "AGENTS.md"


class RuntimeContractAstTest(unittest.TestCase):
    def test_agents_documents_4090_screen_runtime_contract(self):
        self.assertTrue(AGENTS_FILE.exists(), "AGENTS.md must exist")
        text = AGENTS_FILE.read_text(encoding="utf-8")

        self.assertIn("4090 screen runtime contract", text)
        self.assertIn("PYTHONUNBUFFERED=1", text)
        self.assertIn("run_three_station_yearly_protocol.py", text)
        self.assertIn("smoke / pilot / formal", text)
        self.assertIn("pilot-medium", text)
        self.assertIn("PRETRAIN_EPOCHS=500", text)
        self.assertIn("FEW_SHOT_EPOCHS=10", text)
        self.assertIn("PRETRAIN_EPOCHS=2000", text)
        self.assertIn("FEW_SHOT_EPOCHS=20", text)
        self.assertIn("PRETRAIN_EPOCHS=35000", text)
        self.assertIn("PROPOSED_META_EPOCHS=30000", text)
        self.assertIn("FEW_SHOT_EPOCHS=50", text)
        self.assertIn("前 10 个 epoch 每轮打印", text)
        self.assertIn("已收敛", text)


if __name__ == "__main__":
    unittest.main()
