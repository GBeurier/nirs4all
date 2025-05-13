# tests/test_folder.py

import pytest
from nirs4all.data_splitters._folder import get_splitter
from nirs4all.data_splitters._splitter import (
    SystematicCircularSplitter,
)
from sklearn.model_selection import KFold


def test_get_splitter_with_instance():
    splitter_instance = KFold(n_splits=5)
    config = splitter_instance
    splitter = get_splitter(config)
    assert splitter is splitter_instance


def test_get_splitter_with_object_with_split():
    class CustomSplitter:
        def split(self, X, y=None, groups=None):
            pass

    splitter_instance = CustomSplitter()
    config = splitter_instance
    splitter = get_splitter(config)
    assert splitter is splitter_instance


def test_get_splitter_with_known_method():
    config = {'method': 'KFold', 'params': {'n_splits': 5}}
    splitter = get_splitter(config)
    assert isinstance(splitter, KFold)
    assert splitter.n_splits == 5


def test_get_splitter_with_custom_splitter_class():
    config = {'method': SystematicCircularSplitter, 'params': {'test_size': 0.2}}
    splitter = get_splitter(config)
    assert isinstance(splitter, SystematicCircularSplitter)
    assert splitter.test_size == 0.2


def test_get_splitter_with_custom_splitter_instance():
    splitter_instance = SystematicCircularSplitter(test_size=0.2)
    config = {'method': splitter_instance}
    splitter = get_splitter(config)
    assert splitter is splitter_instance


def test_get_splitter_with_invalid_method_type():
    config = {'method': 12345, 'params': {}}
    with pytest.raises(ValueError):
        get_splitter(config)


def test_get_splitter_with_unknown_method_string():
    config = {'method': 'UnknownSplitter', 'params': {}}
    with pytest.raises(ValueError) as exc_info:
        get_splitter(config)
    assert "Invalid splitter method" in str(exc_info.value)


def test_get_splitter_with_invalid_class():
    class NotASplitter:
        pass

    config = {'method': NotASplitter, 'params': {}}
    with pytest.raises(ValueError) as exc_info:
        get_splitter(config)
    assert "Provided class is not a subclass of BaseCrossValidator" in str(exc_info.value)


def test_get_splitter_with_import_path():
    # Assuming 'sklearn.model_selection.KFold' is a valid import path
    config = {'method': 'sklearn.model_selection.KFold', 'params': {'n_splits': 4}}
    splitter = get_splitter(config)
    assert isinstance(splitter, KFold)
    assert splitter.n_splits == 4


def test_get_splitter_with_invalid_import_path():
    config = {'method': 'nonexistent.module.Splitter', 'params': {}}
    with pytest.raises(ValueError) as exc_info:
        get_splitter(config)
    assert "Invalid splitter method" in str(exc_info.value)
