"""
Example usage of YTransformerMixinController for target transformation.

This demonstrates how to use the y_processing keyword to apply
sklearn transformers to target data instead of features.
"""

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from nirs4all.pipeline.config import PipelineConfig
from nirs4all.pipeline.runner import PipelineRunner

# Example pipeline configuration using y_processing
pipeline_config = {
    "steps": [
        {
            "name": "standardize_targets",
            "y_processing": StandardScaler(),  # This will trigger YTransformerMixinController
            "description": "Standardize target values to zero mean and unit variance"
        },
        {
            "name": "scale_targets",
            "y_processing": MinMaxScaler(feature_range=(0, 1)),  # Another y transformation
            "description": "Scale targets to [0, 1] range"
        },
        # You could add regular feature processing steps here too
        {
            "name": "robust_targets",
            "y_processing": RobustScaler(),  # Robust scaling for outliers
            "description": "Apply robust scaling to targets"
        }
    ]
}

def example_y_transformation_usage():
    """
    Example showing how the YTransformerMixinController works:

    1. First step: StandardScaler is applied to targets
       - Context "y" becomes "numeric_StandardScaler1"
       - Targets are standardized (mean=0, std=1)

    2. Second step: MinMaxScaler is applied to the standardized targets
       - Context "y" becomes "numeric_StandardScaler1_MinMaxScaler2"
       - Targets are now in [0, 1] range

    3. Third step: RobustScaler is applied
       - Context "y" becomes "numeric_StandardScaler1_MinMaxScaler2_RobustScaler3"
       - Final robust-scaled targets

    The controller will:
    - Fit each transformer only on training targets
    - Transform all targets (train + test)
    - Update the context["y"] to point to the new processing
    - Store fitted transformers for potential inverse transformation
    """

    # Create pipeline from config
    config = PipelineConfig(pipeline_config, "y_transformation_example")
    runner = PipelineRunner()

    # Assuming you have a dataset with targets loaded:
    # result_dataset, history, pipeline = runner.run(config, your_dataset)

    # The dataset.targets will now have multiple processings:
    # - "raw": Original target data
    # - "numeric": Automatically created numeric conversion
    # - "numeric_StandardScaler1": After first transformation
    # - "numeric_StandardScaler1_MinMaxScaler2": After second transformation
    # - "numeric_StandardScaler1_MinMaxScaler2_RobustScaler3": Final result

    # You can access any processing level:
    # final_targets = dataset.targets.get_targets("numeric_StandardScaler1_MinMaxScaler2_RobustScaler3")
    # standardized_only = dataset.targets.get_targets("numeric_StandardScaler1")

    # And transform predictions back:
    # predictions_in_original_space = dataset.targets.transform_predictions(
    #     model_predictions,
    #     from_processing="numeric_StandardScaler1_MinMaxScaler2_RobustScaler3",
    #     to_processing="numeric"
    # )

    print("Example pipeline configuration created!")
    print("Use this with a dataset that has targets to see YTransformerMixinController in action.")

if __name__ == "__main__":
    example_y_transformation_usage()