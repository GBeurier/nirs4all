# pipeline/operation.py
from nirs4all.core import RUNNER_REGISTRY

class PipelineOperation:
    """Class to represent a pipeline operation that can execute a specific operator."""
    def __init__(self, operator=None, *, keyword: str | None = None, params=None):
        self.operator = operator
        self.keyword = keyword
        self.params = params or {}
        self.runner = self._select_operator()

    def _select_operator(self):
        for runner_cls in RUNNER_REGISTRY:
            if runner_cls.matches(self.operator, self.keyword):
                return runner_cls()
        raise TypeError(f"No matching operator found for {self.operator} with keyword {self.keyword}. Available operators: {[cls.__name__ for cls in RUNNER_REGISTRY]}")

    def execute(self, dataset, context):
        """ Execute the operation using the selected operator runner."""
        return self.runner.run(self.operator, self.params, context, dataset)



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