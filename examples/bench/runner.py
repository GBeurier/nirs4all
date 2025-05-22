from typing import Any, Dict, List, Callable, Union
import logging

from sklearn.base import TransformerMixin, ClusterMixin
import tensorflow as tf
from sklearn.base import BaseEstimator
from torch import nn

logger = logging.getLogger(__name__)


# PipelineDef = List[Union[Callable[[Dict[str, Any]], Dict[str, Any]], Dict[str, Any], List[Any]]]


def run_model(step, data, context=""):
    run(step["model"], data, context + "  ")
    run(step["y_pipeline"], data, context + "  ")
    if "train_params" in step:
        print(context + "  > ", end="")
        print(f"Training: {step['train_params']}")
    
    if "finetune_params" in step:
        print(context + "  > ", end="")
        print(f"Finetuning: {step['finetune_params']}")


def run(step, data, context=""):
        
    print(context, end="")
    if isinstance(step, str):
        if step == "uncluster":
            print(f"Control step: {step}")
        
        else:
            print(f"Unknown step: {step}")
            
        return

    if isinstance(step, dict) and len(step) == 1:
        key, spec = next(iter(step.items()))
        print(f"Control step: {key}")
        if key == "cluster" or key == "C":
            run(step["cluster"], data, context + "  ")
        elif key == "sample_augmentation" or key == "samples" or key == "S":
            for transform in spec:
                run(transform, data, context + "  ")
        elif key == "feature_augmentation" or key == "features" or key == "F":
            for transform in spec:
                run(transform, data, context + "  ")
        elif key == "branch":
            for branch in spec:
                run(branch, data, context + "  ")
        elif key == "stack":
            run_model(spec, data, context + "  ")
            for model in spec["base_learners"]:
                run_model(model, data, context + "  ")
        else:
            print(f"Unknown step: {key}")
            
        return
    
    if isinstance(step, list):
        print("Sub-pipeline")
        for sub_step in step:
            run(sub_step, data, context + "  ")
        
        return
            
    if hasattr(step, "transform") and issubclass(step.__class__, TransformerMixin):
        print(f"Transforming with {step}")
        
    elif hasattr(step, "split"):
        print(f"Splitting with {step.__class__.__name__}")
        
    elif hasattr(step, "fit") and issubclass(step.__class__, ClusterMixin):
        print(f"Clustering with {step.__class__.__name__}")
        
    elif issubclass(step.__class__, BaseEstimator):
        print(f"sklearn > {step.__class__.__name__}")
        
    elif issubclass(step.__class__, tf.keras.Model):
        print(f"tensorflow > {step.__class__.__name__}")
        
    elif issubclass(step.__class__, nn.Module):
        print(f"pytorch > {step.__class__.__name__}")
    
    else:
        print(f"Unknown step: {step}")

