# mypy: ignore-errors
"""
Training script for async TP native vs pipeline decision tree heuristic.

Inherits from AHTrainDecisionTree to leverage the standard AutoHeuristic
training pipeline: grid search, unsafe leaf detection, confidence thresholds,
and artifact code generation.

Usage:
    cd torchgen/_autoheuristic/async_tp
    python train_decision_async_tp_native.py async_tp_native_h100_data.txt --heuristic-name AsyncTPNativeH100
"""

import sys
from pathlib import Path


sys.path.append(str(Path(__file__).absolute().parents[1]))

from train_decision import AHTrainDecisionTree


class AHTrainDecisionTreeAsyncTPNative(AHTrainDecisionTree):
    def __init__(self):
        super().__init__()

    def add_new_features(self, results):
        """
        Features m, k, n, m_local, arith_intensity, m_times_k, m_times_n, k_times_n
        are already computed in the data by convert_native_vs_pipeline_data.py.
        No additional feature engineering needed.
        """
        return (results, [])

    def get_default_config(self, row):
        """Default config when AutoHeuristic returns 'unsure'.
        Pipeline is the safe default — it always works and is near-optimal."""
        return "pipeline"

    def get_allowed_wrong_prediction_pct(self):
        """
        Native vs pipeline has a smaller performance gap than fuse/no_fuse,
        so we can tolerate a slightly higher error rate.
        """
        return 0.02

    def get_test_and_val_size(self):
        """
        Our dataset is small (~40 unique configs × 2 choices = 80 rows),
        so use a larger validation set proportion.
        """
        return (0.15, 0.20)

    def is_unsafe_leaf(self, row, predicted_config, choice2time):
        """
        Mark a leaf as unsafe if the predicted choice is significantly slower
        than the best choice.

        If the leaf is unsafe, the heuristic returns None (unsure), and the
        caller falls back to pipeline (safe default).
        """
        if predicted_config not in choice2time:
            return False

        predicted_time = choice2time[predicted_config]
        best_time = min(choice2time.values())

        # If predicted choice is >10% slower than best, mark unsafe
        if predicted_time > 1.10 * best_time:
            return True

        # Also unsafe if we predict native but pipeline is better by >5%
        if predicted_config == "native" and "pipeline" in choice2time:
            pipeline_time = choice2time["pipeline"]
            if predicted_time > 1.05 * pipeline_time:
                return True

        return False

    def get_grid_search_values(self):
        """
        Grid search over hyperparameters. Use a wider depth range since our
        decision boundary involves multiple features.
        """
        return {
            "max_depth": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 5, 0.05],
            "criterion": ["gini", "entropy"],
        }


if __name__ == "__main__":
    train = AHTrainDecisionTreeAsyncTPNative()
    train.generate_heuristic()
