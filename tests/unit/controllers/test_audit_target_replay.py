"""Target replay must preserve labels and inversion for unlabeled inputs."""

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from nirs4all.data.dataset import SpectroDataset
from nirs4all.pipeline.config.context import ExecutionContext, MapArtifactProvider, RuntimeContext
from nirs4all.pipeline.steps.step_runner import StepRunner


@pytest.mark.parametrize("mode", ["predict", "explain"])
@pytest.mark.parametrize("n_labels", [0, 2, 4])
@pytest.mark.parametrize("chained", [False, True])
def test_saved_target_replay_preserves_rows_and_inverse(mode, n_labels, chained):
    """Reload never refits targets or replaces available labels with placeholders."""
    training_y = np.array([[1.25, 10.5], [2.5, 12.25], [4.75, 16.75]])
    scalers = [MinMaxScaler().fit(training_y)]
    if chained:
        scalers.append(StandardScaler().fit(scalers[0].transform(training_y)))
    provider = MapArtifactProvider({
        1: [(f"y_{type(scaler).__name__}_{index + 1}", scaler) for index, scaler in enumerate(scalers)]
    })
    dataset = SpectroDataset("target_replay")
    dataset.add_samples(np.arange(24).reshape(4, 6), {"partition": "test"})
    labels = np.array([[3.75, 13.25], [6.5, 20.25], [8.75, 25.5], [10.5, 30.75]])[:n_labels]
    if n_labels:
        dataset.add_targets(labels)
        dataset._indexer.mark_excluded([1], cascade_to_augmented=False)
    context = ExecutionContext()
    context.selector.partition = None
    context.state.mode = mode
    runtime = RuntimeContext(artifact_provider=provider, step_number=1)
    runner = StepRunner(mode=mode, verbose=0, show_spinner=False)
    with patch.object(MinMaxScaler, "fit", side_effect=AssertionError("Replay must not fit")), \
            patch.object(StandardScaler, "fit", side_effect=AssertionError("Replay must not fit")):
        result = runner.execute({"y_processing": scalers if chained else scalers[0]}, dataset, context, runtime)

    expected = labels.astype(np.float32)
    predicted = np.array([[5.25, 18.5], [7.75, 22.25]])
    scaled_predictions = predicted.copy()
    for scaler in scalers:
        if len(expected):
            expected = scaler.transform(expected)
        scaled_predictions = scaler.transform(scaled_predictions)
    processing = result.updated_context.state.y_processing
    np.testing.assert_allclose(dataset._targets.get_targets(processing), expected, rtol=1e-6)
    selector = {"y": processing}
    np.testing.assert_allclose(dataset.y(selector), expected[[i for i in range(n_labels) if i != 1]], rtol=1e-6)
    np.testing.assert_allclose(
        dataset._targets.transform_predictions(scaled_predictions, processing, "numeric"), predicted, rtol=1e-6,
    )
