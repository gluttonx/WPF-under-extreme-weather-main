import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "DemoModelTraining.py"


class ThreeStationYearlyProtocolAstTest(unittest.TestCase):
    def test_training_script_exposes_yearly_protocol_knobs(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("YEARLY_PROTOCOL_ENABLED", text)
        self.assertIn("YEARLY_PROTOCOL_METADATA_PATH", text)
        self.assertIn('three_station_yearly_protocol_data/three_station_yearly_protocol_metadata.json', text)
        self.assertIn("def load_yearly_protocol_metadata(", text)
        self.assertIn("yearly_protocol_metadata", text)

    def test_training_script_keeps_extreme_support_and_test_payloads_separate(self):
        text = TRAIN_FILE.read_text(encoding="utf-8")

        self.assertIn("p_test_extre_st", text)
        self.assertIn("nwp_test_extre_st", text)
        self.assertIn("'p_test_extre': p_test_extre_st", text)
        self.assertIn("'nwp_test_extre': nwp_test_extre_st", text)
        self.assertIn("if (SEASONAL_PROTOCOL_ENABLED or YEARLY_PROTOCOL_ENABLED)", text)


if __name__ == "__main__":
    unittest.main()
