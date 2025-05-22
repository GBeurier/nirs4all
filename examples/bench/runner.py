from typing import Any, Dict, List, Callable, Union
import logging

from sklearn.base import TransformerMixin, ClusterMixin
import tensorflow as tf
from sklearn.base import BaseEstimator
from torch import nn

logger = logging.getLogger(__name__)


# PipelineDef = List[Union[Callable[[Dict[str, Any]], Dict[str, Any]], Dict[str, Any], List[Any]]]


class PipelineRunner:
    def __init__(self):
        self.status = None

    def run_pipeline(self, pipeline, data, prefix=""):
        self.status = "running"
        print("Running pipeline")
        context = {
            "branch": 0,
            "group_filter": False,
            "processing": None,
            "augmentation": None,
        }
        for step in pipeline:
            self.run(step, data, context, prefix=prefix)
        self.status = "done"


    def run(self, step, data, context, prefix=""):
        def p(msg, end="\n"):
            print(prefix + msg, end=end)

        if isinstance(step, str):
            if step == "uncluster":
                context["group_filter"] = False
                p(f"Control step: {step}")
            else:
                p(f"Unknown string step: {step}")
            return

        if isinstance(step, dict) and len(step) == 1:
            key, spec = next(iter(step.items()))
            p(f"Control step: {key}")

            #  CLUSTERING
            if key == "cluster":
                self.run_clustering(spec, data, context, prefix=prefix + "  ")

            #  SAMPLE AUGMENTATION
            elif key == "sample_augmentation" or key == "samples" or key == "S":
                for transform in spec:
                    if transform is None:
                        p("  Identity")
                    else:
                        self.run(transform, data, context, prefix=prefix + "  ")

            #  FEATURE AUGMENTATION
            elif key == "feature_augmentation" or key == "features" or key == "F":
                for transform in spec:
                    # pop X from data
                    # fit on X_train
                    # transform X
                    # add X
                    self.run(transform, data, context, prefix=prefix + "  ")

            #  BRANCHING
            elif key == "branch":
                for branch in spec:
                    self.run(branch, data, context, prefix=prefix + "  ")
                    context["branch"] += 1            #  STACKING
            elif key == "stack":
                self.run_model(spec, data, context, prefix=prefix + "  ")
                for model in spec["base_learners"]:
                    self.run_model(model, data, context, prefix=prefix + "  ")
            else:
                p(f"Unknown step: {key}")
            return

        if isinstance(step, dict) and len(step) > 1:
            if "model" in step:
                self.run_model(step, data, context, prefix=prefix)
            elif "cluster" in step:
                self.run_clustering(step, data, context, prefix=prefix + "  ")
            else:
                p(f"Unknown step: {step}")
            return

        if isinstance(step, list):
            p("Sub-pipeline")
            for sub_step in step:
                self.run(sub_step, data, context, prefix=prefix + "  ")
            return

        # FALL BACK NO PARAMS
        if hasattr(step, "transform") and issubclass(step.__class__, TransformerMixin):
            p(f"Transforming with {step}")

        elif hasattr(step, "split"):
            p(f"Splitting with {step.__class__.__name__}")

        elif hasattr(step, "fit") and issubclass(step.__class__, ClusterMixin):
            p(f"Clustering with {step.__class__.__name__}")

        elif issubclass(step.__class__, BaseEstimator):
            p(f"sklearn > {step.__class__.__name__}")

        elif issubclass(step.__class__, tf.keras.Model):
            p(f"tensorflow > {step.__class__.__name__}")

        elif issubclass(step.__class__, nn.Module):
            p(f"pytorch > {step.__class__.__name__}")

        else:
            p(f"Unknown step: {step}")

        
        
    def run_model(self, step, data, context, prefix=""):
        print(prefix + "Model step:")
        self.run(step["model"], data, context, prefix=prefix + "  ")
        self.run(step["y_pipeline"], data, context, prefix=prefix + "  ")
        if "train_params" in step:
            print(prefix + "  > Training: " + str(step['train_params']))
        if "finetune_params" in step:
            print(prefix + "  > Finetuning: " + str(step['finetune_params']))

    def run_clustering(self, step, data, context, prefix=""):
        if hasattr(step, "fit") and issubclass(step.__class__, ClusterMixin):
            print(f"{prefix}Clustering with {step.__class__.__name__}")
            context["group"] = True
        else:
            print(f"{prefix}Unknown clustering step: {step}")
            return

    
