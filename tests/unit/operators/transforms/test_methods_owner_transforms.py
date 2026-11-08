"""Public SDK owner delegation, literal parity and transient handle contracts."""

import builtins
import gc
import pickle
import weakref

import joblib
import numpy as np
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate
from nirs4all.operators.transforms.nirs import savgol

# First two rows copied unchanged from Methods' independent NumPy reference
# parity/fixtures/snv_parameter_contract_v1.json (n4m.snv-parameter-contract.v1).
# No owner result is used to construct the expected arrays.
SNV_INPUT = [[1., 3., 8., 20.], [10., 11., 13., 18.]]
SNV_CENTERED = [[-7., -5., 0., 12.], [-3., -2., 0., 5.]]
SNV_CASES = [
    (True, True, 0, [[-0.9481996460700349, -0.6772854614785964, 0., 1.6254851075486312],
                     [-0.9733285267845753, -0.6488856845230502, 0., 1.6222142113076254]]),
    (True, True, 1, [[-0.8211649813560639, -0.5865464152543313, 0., 1.4077113966103951],
                     [-0.8429272304235246, -0.5619514869490164, 0., 1.404878717372541]]),
    (False, True, 0, [[0.13545709229571928, 0.4063712768871578, 1.0836567383657543, 2.7091418459143854],
                      [3.244428422615251, 3.568871264876776, 4.217756949399826, 5.839971160707452]]),
    (False, True, 1, [[0.11730928305086626, 0.3519278491525988, 0.93847426440693, 2.346185661017325],
                      [2.809757434745082, 3.09073317821959, 3.6526846651686067, 5.057563382541147]]),
    (True, False, 0, SNV_CENTERED), (True, False, 1, SNV_CENTERED),
    (False, False, 0, SNV_INPUT), (False, False, 1, SNV_INPUT),
]


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_mean,with_std,ddof,expected", SNV_CASES)
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_snv_independent_literals(axis, with_mean, with_std, ddof, expected, dtype):
    source = np.asarray(SNV_INPUT, dtype=dtype)
    want = np.asarray(expected, dtype=dtype)
    if axis == 0:
        source, want = source.T, want.T
    before = source.copy()
    actual = StandardNormalVariate(axis=axis, with_mean=with_mean, with_std=with_std, ddof=ddof).transform(source)
    assert actual.shape == source.shape and actual.dtype == source.dtype
    np.testing.assert_array_equal(source, before)
    np.testing.assert_allclose(actual, want, rtol=0, atol=1e-12 if dtype == np.float64 else np.finfo(dtype).eps)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_mean", [False, True])
@pytest.mark.parametrize("with_std", [False, True])
@pytest.mark.parametrize("copy", [False, True])
def test_snv_copy_and_alias_contract(axis, with_mean, with_std, copy):
    source = np.array(SNV_INPUT)[:, ::-1]
    before = source.copy()
    result = StandardNormalVariate(axis=axis, with_mean=with_mean, with_std=with_std, copy=copy).transform(source)
    np.testing.assert_array_equal(source, before)
    alias = not copy and not with_mean and not with_std
    assert (result is source) == alias
    assert bool(np.shares_memory(result, source)) == alias


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("extra", [0, 1])
def test_snv_denominator_boundary_precedes_older_owner(axis, extra, monkeypatch):
    from n4m.transform import scatter

    def forbidden(**_params):
        raise AssertionError("Undefined scaling must fail before entering an older owner")

    monkeypatch.setattr(scatter, "SNV", forbidden)
    source = np.array(SNV_INPUT)
    with pytest.raises(ValueError, match="ddof"):
        StandardNormalVariate(axis=axis, ddof=source.shape[axis] + extra).transform(source)


@pytest.mark.parametrize("axis", [0, 1])
def test_snv_unused_denominator_remains_admitted(axis):
    source, expected = np.array(SNV_INPUT), np.array(SNV_CENTERED)
    if axis == 0:
        source, expected = source.T, expected.T
    actual = StandardNormalVariate(axis=axis, with_std=False, ddof=source.shape[axis] + 1).transform(source)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("with_mean", [False, True])
@pytest.mark.parametrize("with_std", [False, True])
def test_snv_constants_use_mathematical_expectation(with_mean, with_std):
    source = np.full((2, 7), 0.1)
    actual = StandardNormalVariate(with_mean=with_mean, with_std=with_std, ddof=1).transform(source)
    np.testing.assert_array_equal(actual, np.zeros_like(source) if with_mean else source)


def test_snv_axis_zero_remains_stateless_across_batch_sizes():
    operator = StandardNormalVariate(axis=0).fit(np.array(SNV_INPUT))
    np.testing.assert_array_equal(operator.transform(np.array(SNV_INPUT)[:1]), np.zeros((1, 4)))
    np.testing.assert_allclose(operator.transform(np.array(SNV_INPUT)), [[-1., -1., -1., 1.], [1., 1., 1., -1.]], rtol=0, atol=1e-12)


def test_snv_forwards_sdk_truthiness_and_transpose(monkeypatch):
    from n4m.transform import scatter

    calls = []

    class Owner:
        def __init__(self, **params):
            calls.append(params)

        def transform(self, values):
            calls.append(values)
            return values.astype(np.float64)

    monkeypatch.setattr(scatter, "SNV", Owner)
    source = np.array(SNV_INPUT, dtype=np.float32)
    result = StandardNormalVariate(axis=0, with_mean=2, with_std=[], ddof=1, copy=False).transform(source)
    assert calls[0] == {"with_mean": True, "with_std": False, "ddof": 1}
    assert np.shares_memory(calls[1], source)
    np.testing.assert_array_equal(calls[1], source.T)
    assert result.dtype == source.dtype


def test_savgol_forwards_last_axis_and_fixed_interp(monkeypatch):
    from n4m.transform import smoothing

    calls = []

    class Owner:
        def __init__(self, **params):
            calls.append(params)

        def transform(self, values):
            calls.append(values.copy())
            return values.astype(np.float64)

    monkeypatch.setattr(smoothing, "SavitzkyGolay", Owner)
    source = np.arange(72, dtype=np.float32).reshape(2, 3, 12)[..., ::2]
    before = source.copy()
    actual = savgol(source, window_length=4, polyorder=2, deriv=1, delta=0.25)
    assert calls[0] == {"window_length": 4, "polyorder": 2, "deriv": 1, "delta": 0.25, "mode": "interp", "cval": 0.0}
    np.testing.assert_array_equal(calls[1], source.reshape(6, 6))
    np.testing.assert_array_equal(actual, source)
    np.testing.assert_array_equal(before, source)
    assert actual.shape == source.shape and actual.dtype == source.dtype


@pytest.mark.parametrize("cls,params,public_module,owner_name", [
    (StandardNormalVariate, {"ddof": 1}, "scatter", "SNV"),
    (SavitzkyGolay, {"window_length": 10, "polyorder": 3, "deriv": 1, "delta": 0.25}, "smoothing", "SavitzkyGolay"),
])
def test_clone_set_params_pickle_joblib_and_owner_lifetime(cls, params, public_module, owner_name, monkeypatch, tmp_path):
    from n4m.transform import scatter, smoothing

    module = {"scatter": scatter, "smoothing": smoothing}[public_module]
    real_owner = getattr(module, owner_name)
    references = []

    def tracked(**kwargs):
        owner = real_owner(**kwargs)
        references.append(weakref.ref(owner))
        return owner

    monkeypatch.setattr(module, owner_name, tracked)
    operator = cls()
    original_state = vars(operator).copy()
    # The historical SDK instance state contains only constructor attributes.
    restored = cls.__new__(cls)
    restored.__dict__.update(original_state)
    restored = pickle.loads(pickle.dumps(restored))
    assert restored.get_params() == operator.get_params()
    source = np.arange(48, dtype=np.float64).reshape(3, 16)
    operator.transform(source)
    operator.set_params(**params)
    duplicate = clone(operator)
    assert duplicate.get_params() == operator.get_params()
    expected = operator.transform(source)
    np.testing.assert_array_equal(duplicate.fit(source).transform(source), expected)
    destination = tmp_path / "transform.joblib"
    joblib.dump(operator, destination)
    np.testing.assert_array_equal(joblib.load(destination).transform(source), expected)
    assert set(vars(operator)) == set(original_state)
    gc.collect()
    assert references and all(reference() is None for reference in references)


@pytest.mark.parametrize("cls,module_name,owner_name", [(StandardNormalVariate, "scatter", "SNV"), (SavitzkyGolay, "smoothing", "SavitzkyGolay")])
def test_owner_error_releases_transient_object(cls, module_name, owner_name, monkeypatch):
    from n4m.transform import scatter, smoothing

    references = []

    class Broken:
        def __init__(self, **_params):
            references.append(weakref.ref(self))

        def transform(self, _source):
            raise RuntimeError("owner transform failed")

    monkeypatch.setattr({"scatter": scatter, "smoothing": smoothing}[module_name], owner_name, Broken)
    with pytest.raises(RuntimeError, match="owner transform failed"):
        cls().transform(np.arange(16.).reshape(1, 16))
    gc.collect()
    assert all(reference() is None for reference in references)


@pytest.mark.parametrize("cls,module", [(StandardNormalVariate, "n4m.transform.scatter"), (SavitzkyGolay, "n4m.transform.smoothing")])
@pytest.mark.parametrize("error", [ImportError, OSError])
def test_owner_unavailability_propagates(cls, module, error, monkeypatch):
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name == module:
            raise error("selected owner unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    with pytest.raises(error, match="selected owner unavailable"):
        cls().transform(np.arange(16.).reshape(1, 16))


def test_existing_sparse_finite_axis_and_fit_contracts():
    matrix = sparse.csr_matrix(np.ones((2, 16)))
    for method in ("fit", "transform"):
        with pytest.raises(TypeError, match="sparse"):
            getattr(StandardNormalVariate(), method)(matrix)
        with pytest.raises(ValueError, match="[Ss]parse"):
            getattr(SavitzkyGolay(), method)(matrix)
        with pytest.raises(ValueError):
            getattr(StandardNormalVariate(), method)(np.full((2, 16), np.nan))
        with pytest.raises(ValueError, match="axis"):
            getattr(StandardNormalVariate(axis=2), method)(np.ones((2, 16)))
    source = np.arange(32.).reshape(2, 16)
    pipeline = Pipeline([("snv", StandardNormalVariate()), ("sg", SavitzkyGolay())])
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(pipeline)).fit_transform(source), pipeline.fit_transform(source))


# Independent SciPy 1.17.1 literals from Root's frozen helper:
# ceab07b3ffcae1ea8678a56b6d2f68097b332a5fd0dc24aeb89f1df29775d5b5
# Same exact input/calls as the Methods SG contract tests; no runtime SciPy oracle.
SG_INPUT = [
    0.5, 1.25, -0.75, 3.0, 2.5, -1.0, 4.25, 0.0,
    5.5, 2.0, -2.25, 6.0, 1.5, 3.75, -0.5, 7.0,
]
SG_CASES = [
    # scipy.signal.savgol_filter(np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0, axis=-1, mode="interp", cval=0.0)
    pytest.param(10, 3, 0, 1.0, [
        0.6000000000000022, 0.7136363636363636, 0.9181818181818171, 1.1909090909090896,
        1.5090909090909084, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
        2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 1.681002331002326,
        1.816200466200459, 2.3671911421911345, 3.450815850815845, 5.183916083916084,
    ], id="even_interp"),
    # scipy.signal.savgol_filter(np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=1, delta=0.25, axis=-1, mode="interp", cval=0.0)
    pytest.param(10, 3, 1, 0.25, [
        0.24242424242423105, 0.6515151515151453, 0.9696969696969675, 1.1969696969696977,
        1.3333333333333357, 2.894036519036521, -1.6588966588966545, 0.8190559440559428,
        -0.11344211344211136, 0.01418026418026308, -0.6744852369852319, -0.13500388500389632,
        1.294483294483288, 3.1913364413364427, 5.555555555555569, 8.387140637140666,
    ], id="even_derivative"),
    # scipy.signal.savgol_filter(np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=1.0, axis=-1, mode="interp", cval=0.0)
    pytest.param(10, 3, 0, 1.0, [
        0.6000000000000022, 0.7136363636363636, 0.9181818181818171, 1.1909090909090896,
        1.5090909090909084, 1.8499999999999996, 2.1909090909090914, 2.50909090909091,
        2.7818181818181826, 2.986363636363636,
    ], id="even_minimum_width"),
    # scipy.signal.savgol_filter(np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=0, delta=0.0, axis=-1, mode="interp", cval=0.0)
    pytest.param(10, 3, 0, 0.0, [
        0.6000000000000022, 0.7136363636363636, 0.9181818181818171, 1.1909090909090896,
        1.5090909090909084, 2.4656250000000024, 1.987500000000021, 1.9468750000000061,
        2.0687500000000147, 2.8000000000000105, 1.7328125000000134, 1.681002331002326,
        1.816200466200459, 2.3671911421911345, 3.450815850815845, 5.183916083916084,
    ], id="zero_delta_smoothing"),
    # scipy.signal.savgol_filter(np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=11, polyorder=3, deriv=0, delta=1.0, axis=-1, mode="interp", cval=0.0)
    pytest.param(11, 3, 0, 1.0, [
        1.0489510489510474, 0.41433566433566105, 0.43181818181817677, 0.8916083916083842,
        1.5839160839160753, 2.298951048951052, 1.8141025641025799, 2.221445221445235,
        1.8846153846153932, 2.4959207459207504, 2.1853146853147076, 1.737762237762254,
        1.5891608391608565, 1.998251748251762, 3.2237762237762317, 5.52447552447552,
    ], id="odd_interp"),
    # scipy.signal.savgol_filter(np.array([0.5,1.25,-0.75,3,2.5,-1,4.25,0,5.5,2,-2.25,6,1.5,3.75,-0.5,7], dtype=np.float64),
    #     window_length=10, polyorder=3, deriv=4, delta=1.0, axis=-1, mode="interp", cval=0.0)
    pytest.param(10, 3, 4, 1.0, [
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ], id="derivative_above_polyorder"),
]


@pytest.mark.parametrize("window,degree,deriv,delta,expected", SG_CASES)
@pytest.mark.parametrize("ndim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float16, np.float32, np.float64])
def test_savgol_independent_literals_shape_dtype_and_copy(window, degree, deriv, delta, expected, ndim, dtype):
    source = np.array(SG_INPUT[:len(expected)], dtype=dtype)
    shape = (2,) * (ndim - 1) + (len(expected),)
    source = np.broadcast_to(source, shape)
    # Preserve literal sample values while giving the adapter a strided last axis.
    source = np.repeat(source, 2, axis=-1)[..., ::2]
    before = source.copy()
    actual = savgol(source, window, degree, deriv, delta)
    operator = SavitzkyGolay(window, degree, deriv, delta, copy=False)
    from_class = operator.fit(source).transform(source, copy=False)
    output_dtype = np.float32 if dtype == np.float32 else np.float64
    want = np.broadcast_to(np.asarray(expected, dtype=output_dtype), shape)
    assert actual.shape == shape and actual.dtype == output_dtype
    np.testing.assert_allclose(actual, want, atol=1e-9, rtol=1e-10)
    np.testing.assert_array_equal(actual, from_class)
    np.testing.assert_array_equal(before, source)
    assert not np.shares_memory(actual, source)


def test_savgol_short_signal_and_zero_spacing_derivative_refusals():
    from n4m import N4MError

    with pytest.raises(ValueError, match="window_length"):
        savgol(np.asarray(SG_INPUT[:9]), window_length=10, polyorder=3)
    # SciPy 1.17.1 with the same input, (10,3), deriv=1 and delta=0
    # raises ZeroDivisionError. The selected owner reports INVALID_ARGUMENT.
    with pytest.raises(N4MError) as caught:
        savgol(np.asarray(SG_INPUT), window_length=10, polyorder=3, deriv=1, delta=0.0)
    assert caught.value.status_name == "INVALID_ARGUMENT"

