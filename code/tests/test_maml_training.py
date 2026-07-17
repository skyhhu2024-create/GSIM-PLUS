import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


COMMON = Path(__file__).resolve().parents[1] / "00_common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))

import gsim_core as core
from gsim_plus_config import RANDOM_SEED


def synthetic_anchor_data():
    rng = np.random.default_rng(123)
    anchors = {}
    similarity_rows = []
    month = np.tile(np.arange(1, 13), 4)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    for index in range(6):
        x = rng.normal(size=(48, 3))
        y = 0.7 * x[:, 0] - 0.25 * x[:, 1] + 0.15 * x[:, 2]
        station_id = f"A{index}"
        anchors[station_id] = {
            "X": x,
            "y": y,
            "y_original": y,
            "valid": np.ones(48, dtype=bool),
            "month_sin": month_sin,
            "month_cos": month_cos,
        }
        similarity_rows.append(
            {
                "target_station": f"T{index}",
                "anchor_station": station_id,
                "similarity": 0.9,
            }
        )
    return anchors, pd.DataFrame(similarity_rows)


class MAMLTrainingTests(unittest.TestCase):
    def test_prediction_constraint_is_nonnegative_and_feedback_consistent(self):
        pred_std, pred_orig = core.constrain_nonnegative_prediction(-3.0, flow_mean=1.0, flow_std=2.0)

        self.assertEqual(pred_orig, 0.0)
        self.assertEqual(pred_std, -0.5)

    def test_meta_training_updates_parameters_deterministically(self):
        anchors, similarity = synthetic_anchor_data()
        config = {
            "meta_lr": 0.001,
            "inner_lr": 0.05,
            "inner_steps": 2,
            "meta_batch_size": 4,
            "epochs": 3,
            "hidden_dim": 16,
            "input_dim": 3,
            "first_order": True,
            "meta_scenarios": ["random30", "block3", "block6", "block12", "block25plus"],
        }

        core.set_global_seed(RANDOM_SEED)
        initial = core.MAMLModel(input_dim=3, hidden_dim=16)
        initial_state = {key: value.detach().clone() for key, value in initial.state_dict().items()}

        trained_1 = core.train_maml_model(anchors, similarity, None, config)
        trained_2 = core.train_maml_model(anchors, similarity, None, config)

        max_change = max(
            (trained_1.state_dict()[key].cpu() - initial_state[key].cpu()).abs().max().item()
            for key in initial_state
        )
        max_repeat_difference = max(
            (trained_1.state_dict()[key].cpu() - trained_2.state_dict()[key].cpu()).abs().max().item()
            for key in trained_1.state_dict()
        )

        self.assertGreater(max_change, 0.0)
        self.assertLess(max_repeat_difference, 1e-7)

    def test_hidden_target_truth_does_not_change_validation_inputs(self):
        n_months = 48
        dates = pd.date_range("2000-01-31", periods=n_months, freq=pd.offsets.MonthEnd())
        month = dates.month.to_numpy()
        values = 20.0 + 4.0 * np.sin(2 * np.pi * month / 12)
        hidden = np.array([10, 11, 12])

        def build(shift):
            shifted = values.copy()
            shifted[hidden] += shift
            return {
                "dates": dates.to_numpy(),
                "year": dates.year.to_numpy(),
                "month": month,
                "month_sin": np.sin(2 * np.pi * month / 12),
                "month_cos": np.cos(2 * np.pi * month / 12),
                "y_original": shifted,
                "valid": np.ones(n_months, dtype=bool),
            }

        entry_a = core.build_validation_entry("T", build(0.0), hidden)
        entry_b = core.build_validation_entry("T", build(10000.0), hidden)

        self.assertAlmostEqual(entry_a["flow_mean"], entry_b["flow_mean"])
        self.assertAlmostEqual(entry_a["flow_std"], entry_b["flow_std"])
        self.assertTrue(np.isnan(entry_a["std_series_init"][hidden]).all())
        self.assertTrue(np.isnan(entry_b["std_series_init"][hidden]).all())


if __name__ == "__main__":
    unittest.main()
