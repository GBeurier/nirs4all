"""
JAX Model Wrapper - Wrapper for Flax models to support pickling and prediction.
"""
import numpy as np
from typing import Any, Dict

class JaxModelWrapper:
    """Wrapper to hold Flax model definition and trained state."""
    def __init__(self, model, state):
        self.model = model
        self.state = state

    def predict(self, X):
        variables = {'params': self.state.params}
        if self.state.batch_stats is not None:
            variables['batch_stats'] = self.state.batch_stats

        logits = self.state.apply_fn(variables, X, train=False)
        return np.array(logits)

    def __getstate__(self):
        # For pickling
        return {'model': self.model, 'state': self.state}

    def __setstate__(self, state):
        self.model = state['model']
        self.state = state['state']
