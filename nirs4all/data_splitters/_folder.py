#folder.py

from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold, ShuffleSplit, GroupKFold, StratifiedShuffleSplit, BaseCrossValidator, TimeSeriesSplit
from typing import Any, Dict
from typing import Any, Dict, Union

from ._splitter import SystematicCircularSplitter, KBinsStratifiedSplitter, KMeansSplitter, KennardStoneSplitter, SPXYSplitter
import importlib

def get_splitter(split_config: Dict[str, Any]) -> BaseCrossValidator:
    """
    Factory function to instantiate cross-validation splitters.
    
    Parameters
    ----------
    split_config : Dict[str, Any]
        Configuration dictionary with 'method' and 'params'.
    
    Returns
    -------
    splitter : BaseCrossValidator
        An instance of a splitter compatible with scikit-learn.
    """
    if isinstance(split_config, BaseCrossValidator):
        return split_config
    
    if isinstance(split_config, object) and hasattr(split_config, 'split'):
        return split_config
    
    method = split_config.get('method')
    params = split_config.get('params', {})

    if isinstance(method, str):
        # Mapping of known scikit-learn splitters
        sklearn_splitters = {
            'KFold': KFold,
            'RepeatedKFold': RepeatedKFold,
            'StratifiedKFold': StratifiedKFold,
            'RepeatedStratifiedKFold': RepeatedStratifiedKFold,
            'ShuffleSplit': ShuffleSplit,
            'StratifiedShuffleSplit': StratifiedShuffleSplit,
            'GroupKFold': GroupKFold,
            'TimeSeriesSplit': TimeSeriesSplit,
            # TODO Add other sklearn splitters
            'SystematicCircularSplitter': SystematicCircularSplitter,
            'KBinsStratifiedSplitter': KBinsStratifiedSplitter,
            'KMeansSplitter': KMeansSplitter,
            'KennardStoneSplitter': KennardStoneSplitter,
            'SPXYSplitter': SPXYSplitter,
        }

        if method in sklearn_splitters:
            splitter_class = sklearn_splitters[method]
            return splitter_class(**params)
        else:
            # Attempt to load a custom splitter via import path
            try:
                module_name, class_name = method.rsplit('.', 1)
                module = importlib.import_module(module_name)
                splitter_class = getattr(module, class_name)
                if issubclass(splitter_class, BaseCrossValidator):
                    return splitter_class(**params)
                else:
                    raise ValueError(f"The class {class_name} is not a subclass of BaseCrossValidator.")
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(f"Invalid splitter method: {method}.") from e
            
    elif isinstance(method, type):
        # If method is a class, instantiate it
        if issubclass(method, BaseCrossValidator):
            return method(**params)
        else:
            raise ValueError("Provided class is not a subclass of BaseCrossValidator.")
    elif isinstance(method, BaseCrossValidator):
        # If method is an instance, return it directly
        return method
    else:
        raise ValueError(f"Invalid method type: {type(method)}. Must be a string, class, or instance.")


def run_splitter(config, dataset):
    """
    Run the splitter on the dataset.
    """
    if config is None:
        return dataset
    
    splitter = get_splitter(config)
    folds = list(splitter.split(dataset.x_train_('concat', disable_augmentation=True), dataset.y_train_(disable_augmentation=True), dataset.group_train))
    return folds
