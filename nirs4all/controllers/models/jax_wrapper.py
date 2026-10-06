"""
JAX Model Wrapper - Wrapper for Flax models to support pickling and prediction.
"""
from typing import Any

import numpy as np


class JaxModelWrapper:
    """Wrapper to hold Flax model definition and trained state."""
    def __init__(self, model, state, is_classification: bool = False):
        self.model = model
        self.state = state
        self.is_classification = is_classification

    def predict(self, X):
        variables = {'params': self.state.params}
        if self.state.batch_stats is not None:
            variables['batch_stats'] = self.state.batch_stats

        logits = self.state.apply_fn(variables, X, train=False)
        return np.array(logits)

    def __getstate__(self):
        # For pickling
        return {'model': self.model, 'state': self.state, 'is_classification': self.is_classification}

    def __setstate__(self, state):
        self.model = state['model']
        self.state = state['state']
        self.is_classification = state.get('is_classification', False)
