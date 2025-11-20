"""
Q17 Nicon Comparison
====================
Comparison of Nicon models across TensorFlow, PyTorch, and JAX.
"""

import argparse
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit

from nirs4all.data import DatasetConfigs
from nirs4all.pipeline import PipelineConfigs, PipelineRunner
from nirs4all.utils.backend import TF_AVAILABLE, TORCH_AVAILABLE, JAX_AVAILABLE

# Import models
models = {}
if TF_AVAILABLE:
    from nirs4all.operators.models.tensorflow.nicon import nicon as nicon_tf
    from nirs4all.operators.models.tensorflow.nicon import nicon_classification as nicon_classification_tf
    models['tensorflow'] = {'reg': nicon_tf, 'clf': nicon_classification_tf}
if TORCH_AVAILABLE:
    from nirs4all.operators.models.pytorch.nicon import nicon as nicon_torch
    from nirs4all.operators.models.pytorch.nicon import nicon_classification as nicon_classification_torch
    models['pytorch'] = {'reg': nicon_torch, 'clf': nicon_classification_torch}
if JAX_AVAILABLE:
    from nirs4all.operators.models.jax.nicon import nicon as nicon_jax
    from nirs4all.operators.models.jax.nicon import nicon_classification as nicon_classification_jax
    models['jax'] = {'reg': nicon_jax, 'clf': nicon_classification_jax}

parser = argparse.ArgumentParser(description='Q17 Nicon Comparison')
parser.add_argument('--plots', action='store_true', help='Show plots interactively')
parser.add_argument('--show', action='store_true', help='Show all plots')
args = parser.parse_args()

print(f"Available frameworks: {list(models.keys())}")

# 1. Regression
print("\n--- Regression Comparison ---")
data_path_reg = 'sample_data/regression'
pipeline_reg = [
    StandardScaler,
    ShuffleSplit(n_splits=1, test_size=0.25, random_state=42),
    ShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
]

for fw, m in models.items():
    pipeline_reg.append({
        'model': m['reg'],
        'train_params': {'epochs': 5, 'verbose': 0}, # Short training for test
        'name': f'nicon_{fw}'
    })

runner_reg = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
preds_reg, _ = runner_reg.run(PipelineConfigs(pipeline_reg), DatasetConfigs(data_path_reg))
print("Regression Results:")
print(preds_reg)


# 2. Binary Classification
print("\n--- Binary Classification Comparison ---")
data_path_bin = 'sample_data/binary'
pipeline_bin = []
pipeline_bin.append(StandardScaler)
pipeline_bin.append(ShuffleSplit(n_splits=1, test_size=0.25, random_state=42))

for fw, m in models.items():
    pipeline_bin.append({
        'model': m['clf'], # nicon_classification defaults to num_classes=2
        'train_params': {'epochs': 5, 'verbose': 0},
        'name': f'nicon_bin_{fw}'
    })

runner_bin = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
preds_bin, _ = runner_bin.run(PipelineConfigs(pipeline_bin), DatasetConfigs(data_path_bin))
print("Binary Classification Results:")
print(preds_bin)


# 3. Multi-class Classification
print("\n--- Multi-class Classification Comparison ---")
data_path_clf = 'sample_data/classification'
pipeline_clf = []
pipeline_clf.append(StandardScaler)
pipeline_clf.append(ShuffleSplit(n_splits=1, test_size=0.25, random_state=42))

for fw, m in models.items():
    clf_func = m['clf']

    # Use dictionary format to pass parameters and avoid serialization issues with partials
    pipeline_clf.append({
        'model': {
            'type': 'function',
            'func': clf_func,
            'params': {'num_classes': 3},
            'framework': fw
        },
        'train_params': {'epochs': 5, 'verbose': 0},
        'name': f'nicon_multi_{fw}'
    })

runner_clf = PipelineRunner(save_files=False, verbose=1, plots_visible=args.plots)
preds_clf, _ = runner_clf.run(PipelineConfigs(pipeline_clf), DatasetConfigs(data_path_clf))
print("Multi-class Classification Results:")
print(preds_clf)

if args.show:
    plt.show()
