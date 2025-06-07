"""DummyController.py - A dummy controller for testing purposes in the nirs4all pipeline."""

from typing import Any, Dict, TYPE_CHECKING

from .operator_controller import OperatorController
from .operator_registry import register_controller

if TYPE_CHECKING:
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.spectra.spectra_dataset import SpectraDataset

@register_controller
class DummyController(OperatorController):
    """Dummy controller for testing purposes."""

    priority = 1000  # Lower priority than other controllers

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:
        """Check if the operator matches the step and keyword."""
        return True  # Always matches for testing

    def execute(
        cls,
        step: Any,
        operator: Any,
        dataset: 'SpectraDataset',
        context: Dict[str, Any],
        runner: 'PipelineRunner'
    ):

        # # raw train branch-2 spectra, 3-D tensor
        # X_train = ds.get_features({"partition":"train", "branch":2}, layout="3d")

        # # fetch targets only for augmented group-4 samples
        # y_aug = ds.y({"group":4, "augmented":True}, columns=["y_bin","float1"])

        # # explicit row IDs still work
        # rows = [0, 7, 9]
        # X_subset = ds.x(rows, layout="2d_concat")




        # 1) fetch + fit
        # batch_tr = ds.x({"partition":"train"})
        # tf_mm    = ds.fit_transformer(batch_tr, MinMaxScaler())

        # # 2) apply everywhere
        # ds.apply_transformer(tf_mm, "all")

        # # 3) gaussian-noise augmentation of augmented==False train rows
        # class GaussianNoise:
        #     def __init__(self, std=0.01): self.std = std
        #     def __call__(self, X): return X + np.random.randn(*X.shape)*self.std

        # ds.augment({"partition":"train","augmented":False},
        #         [GaussianNoise(std=0.02)],
        #         copies=2)




        # splits = KFold(n_splits=5, shuffle=True).split(ds.x({"partition":"train"}))
        # row_splits = [[rows[idx] for idx in fold] for _, fold in splits]
        # ds.set_folds(row_splits)

        # for tr_rows, val_rows in ds.fold_iter(5):
        #     X_tr = ds.x(tr_rows, layout="3d")
        #     ...


        """Run the operator with the given parameters and context."""
        print(f"Executing dummy operation for step: {step}, keyword: {context.get('keyword', '')}")
