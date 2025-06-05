# pipeline/operation.py
from typing import Any

from nirs4all.core import RUNNER_REGISTRY

class PipelineOperation:
    """Class to represent a pipeline operation that can execute a specific operator."""
    def __init__(self, step: Any, operator: Any, keyword: str):
        self.step = step
        self.operator = operator
        self.keyword = keyword
        self.wrapper = self._select_wrapper()

    def _select_wrapper(self):
        for wrapper_cls in RUNNER_REGISTRY:
            if wrapper_cls.matches(self.step):
                return wrapper_cls()
        raise TypeError(f"No matching wrapper found for {self.step}. Available wrappers: {[cls.__name__ for cls in RUNNER_REGISTRY]}")

    def execute(self, dataset, context):
        """ Execute the operation using the selected operator wrapper."""
        return self.wrapper.execute(self.step, dataset, context)

    # def get_name(self):
    #     """Get the name of the operation."""
    #     return self.operator.__class__.__name__ if self.operator else "UnnamedOperation"

    # def __repr__(self):
    #     """String representation of the operation."""
    #     return f"PipelineOperation(operator={self.get_name()}, keyword={self.keyword}, params={self.params})"

    # def __hash__(self) -> int:
    #     return hash((self.get_name(), self.keyword, frozenset(self.params.items()), self.wrapper.__class__.__name__, self.operator.__hash__() if self.operator else None))

    # def save(self, path: str):
    #     """Save the operation to a file."""
    #     import pickle
    #     with open(path, 'wb') as f:
    #         pickle.dump(self, f)

    # @classmethod
    # def load(cls, path: str):
    #     """Load the operation from a file."""
    #     import pickle
    #     with open(path, 'rb') as f:
    #         return pickle.load(f)

# # pipeline/runners/torch_runner.py
# import torch.nn as nn
# from . import register_runner
# from .base import BaseOpRunner

# @register_runner
# class TorchRunner(BaseOpRunner):
#     priority = 30

#     @classmethod
#     def matches(cls, op, keyword):
#         return isinstance(op, nn.Module)

#     def run(self, op, params, context, dataset):
#         # boucle d’apprentissage simplifiée ...
#         ...