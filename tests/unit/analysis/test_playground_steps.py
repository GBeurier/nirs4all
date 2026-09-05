"""Real operators, exact axes and aligned exploration input identities."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from nirs4all.analysis.playground_steps import augment_batch, sample_batch, transform_batch
from nirs4all.analysis.playground_types import PreviewBatch, PreviewLimits
from nirs4all.data.selection.sampling import kmeans_sample, random_sample, stratified_sample
from nirs4all.operators.augmentation.spectral import GaussianAdditiveNoise
from nirs4all.operators.transforms import CropTransformer, Resampler, SavitzkyGolay, StandardNormalVariate


@pytest.fixture
def batch():
    rng = np.random.default_rng(28)
    return PreviewBatch.from_arrays(rng.normal(size=(18, 15)), wavelengths=np.linspace(1100.001, 1250.003, 15),
                                    y=np.arange(18) / 3, sample_ids=[f"s{i}" for i in range(18)],
                                    metadata={"subject": [f"g{i // 3}" for i in range(18)]},
                                    partitions=["train"] * 12 + ["test"] * 6, header_unit="nm")


@pytest.mark.parametrize("method", ["all", "random", "stratified", "kmeans"])
def test_sampling_matches_owner_and_keeps_every_aligned_column(batch, method):
    selected, messages = sample_batch(batch, method=method, n_samples=7, seed=19)
    expected = {"all": lambda: np.arange(18), "random": lambda: random_sample(18, 7, 19),
                "stratified": lambda: stratified_sample(batch.x, batch.y, 7, 19),
                "kmeans": lambda: kmeans_sample(batch.x, 7, 19)}[method]()
    assert messages == []
    for name in ("x", "y", "origins", "sample_ids", "partitions"):
        np.testing.assert_array_equal(getattr(selected, name), getattr(batch, name)[expected])
    np.testing.assert_array_equal(selected.metadata["subject"], batch.metadata["subject"][expected])


def test_missing_targets_stay_missing_and_stratified_policy_is_visible():
    batch = PreviewBatch.from_arrays(np.arange(30).reshape(10, 3))
    sampled, messages = sample_batch(batch, method="stratified", n_samples=4)
    assert sampled.y is None
    assert "no observed y" in messages[0]
    transformed = transform_batch(sampled, StandardScaler())
    assert transformed.y is None
    assert transformed.axis_kind == "feature_index"
    identity = transform_batch(sampled, Resampler())
    np.testing.assert_array_equal(identity.x, sampled.x)
    assert identity.axis_kind == "feature_index"
    assert identity.header_unit is None


def test_snv_then_sg_matches_real_owner_and_never_fits_input_instance(batch):
    snv, sg = StandardNormalVariate(), SavitzkyGolay(window_length=5, polyorder=2)
    result = transform_batch(transform_batch(batch, snv), sg)
    expected = sg.fit_transform(snv.fit_transform(batch.x))
    np.testing.assert_allclose(result.x, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(result.wavelengths, batch.wavelengths)
    np.testing.assert_array_equal(result.origins, batch.origins)
    scaler = StandardScaler()
    transform_batch(batch, scaler)
    assert not hasattr(scaler, "mean_")


def test_resampling_exact_nonrounded_axis_and_crop_chain(batch):
    target = np.linspace(1100.004321, 1250.001234, 21)
    actual = transform_batch(batch, Resampler(target_wavelengths=target))
    expected = Resampler(target_wavelengths=target).fit_transform(batch.x, wavelengths=batch.wavelengths)
    np.testing.assert_array_equal(actual.wavelengths, target)
    np.testing.assert_allclose(actual.x, expected)
    assert actual.header_unit == "nm"
    cropped = transform_batch(actual, CropTransformer(start=2, end=9))
    np.testing.assert_array_equal(cropped.wavelengths, target[2:9])
    np.testing.assert_array_equal(cropped.x, expected[:, 2:9])


def test_component_axis_is_not_mislabeled_as_wavelength(batch):
    result = transform_batch(batch, PCA(n_components=3))
    assert result.axis_kind == "feature_index"
    assert result.header_unit is None
    np.testing.assert_array_equal(result.wavelengths, np.arange(3))


def test_augmentation_values_and_repeated_labels_match_real_owner(batch):
    result, info = augment_batch(batch, GaussianAdditiveNoise(sigma=0.02, random_state=7), copies=2)
    owner = GaussianAdditiveNoise(sigma=0.02, random_state=7)
    expected = np.concatenate([batch.x] + [owner.fit_transform(batch.x) for _ in range(2)])
    np.testing.assert_array_equal(result.x, expected)
    for name in ("y", "sample_ids", "origins", "partitions"):
        np.testing.assert_array_equal(getattr(result, name), np.tile(getattr(batch, name), 3))
    np.testing.assert_array_equal(result.metadata["subject"], np.tile(batch.metadata["subject"], 3))
    assert info == {"original_count": 18, "augmented_count": 36, "total_count": 54}


class SupervisedWitness(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        if y is None:
            raise ValueError("observed target required")
        self.offset_ = np.mean(y)
        return self

    def transform(self, X):
        return X + self.offset_


def test_supervised_transform_receives_observed_y_once(batch):
    result = transform_batch(batch, SupervisedWitness())
    np.testing.assert_array_equal(result.x, batch.x + np.mean(batch.y))
    with pytest.raises(ValueError, match="observed target required"):
        transform_batch(PreviewBatch.from_arrays(batch.x), SupervisedWitness())


def test_expansion_budget_precedes_clone_and_fit(batch):
    class ForbiddenClone:
        def get_params(self, deep=False):
            raise AssertionError("must reject before clone")

    with pytest.raises(ValueError, match="host limits"):
        augment_batch(batch, ForbiddenClone(), copies=10**50, limits=PreviewLimits(max_samples=30))
    with pytest.raises(ValueError, match="host limits"):
        transform_batch(batch, Resampler(target_wavelengths=np.linspace(1100, 1200, 50)), limits=PreviewLimits(max_features=20))
    result, _ = augment_batch(batch, StandardScaler(), copies=1, limits=PreviewLimits(max_samples=36))
    assert result.x.shape == (36, 15)


@pytest.mark.parametrize("value", [True, 1.5, -1, "2"])
def test_invalid_copy_counters_are_not_coerced(batch, value):
    with pytest.raises(ValueError):
        augment_batch(batch, StandardScaler(), copies=value)


def test_batch_alignment_and_sampling_options_reject_typos(batch):
    with pytest.raises(ValueError, match="one value"):
        PreviewBatch.from_arrays(batch.x, y=[1])
    with pytest.raises(ValueError, match="one value"):
        PreviewBatch.from_arrays(batch.x, y=[1], metadata={"y": list(range(18))})
    with pytest.raises(ValueError, match="feature axis"):
        PreviewBatch.from_arrays(batch.x, wavelengths=[1])
    with pytest.raises(ValueError, match="Unknown sampling"):
        sample_batch(batch, method="randmo")
    with pytest.raises(ValueError, match="integer"):
        sample_batch(batch, seed=True)
    with pytest.raises(ValueError, match="out of range"):
        batch.take([-1])


def test_budget_is_checked_before_materializing_declared_input():
    class ForbiddenArray:
        shape = (100, 3)

        def __array__(self, *args, **kwargs):
            raise AssertionError("must admit before converting")

    with pytest.raises(ValueError, match="host limits"):
        PreviewBatch.from_arrays(ForbiddenArray(), limits=PreviewLimits(max_samples=2))
