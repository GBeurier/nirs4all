"""Real optional neural builders and classification objectives."""

import numpy as np
import pytest


@pytest.mark.tensorflow
def test_tensorflow_multiclass_and_default_architecture_shapes():
    pytest.importorskip("tensorflow")
    from importlib import import_module

    nicon = import_module("nirs4all.operators.models.tensorflow.nicon")
    from nirs4all.controllers.models.factory import ModelFactory
    from nirs4all.operators.models.tensorflow.generic import SEResNet_model, inception1D

    for factory in (nicon.decon_Sep, nicon.decon_Sep_classification, nicon.transformer_model,
                    nicon.transformer_model_classification):
        assert ModelFactory.detect_framework(factory) == "tensorflow"
    model = nicon.decon_Sep_classification((256, 1), num_classes=3)
    assert model.output_shape[-1] == 3
    assert model.layers[-1].activation.__name__ == "softmax"
    assert inception1D((256, 1), {}).input_shape[1:] == (256, 1)
    assert SEResNet_model((512, 1), {}).input_shape[1:] == (512, 1)


@pytest.mark.jax
@pytest.mark.parametrize("num_classes", [2, 3])
def test_jax_binary_and_multiclass_training_use_finite_logits_loss(num_classes):
    jax = pytest.importorskip("jax")
    nn = pytest.importorskip("flax.linen")
    optax = pytest.importorskip("optax")
    from nirs4all.controllers.models.jax_model import JaxModelController
    from nirs4all.core.task_type import TaskType
    from nirs4all.operators.models.jax import nicon

    # Every advertised classification factory must end in a linear Dense.
    names = ("decon_classification", "decon_Sep_classification", "nicon_classification",
             "customizable_nicon_classification", "nicon_VG_classification",
             "customizable_decon_classification", "decon_layer_classification",
             "transformer_classification", "transformer_VG_classification", "transformer_model_classification")
    for name in names:
        built = getattr(nicon, name)((256, 1), num_classes=num_classes)
        assert isinstance(built.layers[-1], nn.Dense)
        assert built.layers[-1].features == (1 if num_classes == 2 else num_classes)

    # Use a tiny real Flax model to exercise both jitted train and validation steps.
    model = nicon.DynamicModel(layers=[nn.Dense(features=1 if num_classes == 2 else num_classes)])
    X = np.random.default_rng(14).normal(size=(12, 4)).astype(np.float32)
    y = (np.arange(12) % num_classes).reshape(-1, 1).astype(np.float32)
    controller = JaxModelController()
    task = TaskType.BINARY_CLASSIFICATION if num_classes == 2 else TaskType.MULTICLASS_CLASSIFICATION
    trained = controller._train_model(model, X, y, X_val=X, y_val=y, task_type=task, epochs=2, batch_size=6)
    logits = trained.predict(X)
    loss = (optax.sigmoid_binary_cross_entropy(logits[:, 0], y[:, 0]) if num_classes == 2
            else optax.softmax_cross_entropy_with_integer_labels(logits, y[:, 0].astype(int)))
    assert np.all(np.isfinite(loss))
    probabilities = controller._predict_proba_model(trained, X)
    assert np.all(np.isfinite(probabilities))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(controller._predict_model(trained, X).reshape(-1), probabilities.argmax(axis=1))
