# Model Naming and CV Averaging Refactoring Summary

## Overview

This refactoring addresses the critical issues identified in the AbstractModelController regarding:

1. **Model naming inconsistency** - Model IDs were reconstructed locally without continuity
2. **CV strategy confusion** - Train and finetune logic was mixed in CV strategies
3. **Missing avg/w-avg predictions** - Average predictions were not generated after fold training
4. **Prediction storage problems** - Model naming schema was inconsistent across the pipeline

## Key Changes

### 1. New ModelNamingManager (`model_naming.py`)

- **Centralized model naming**: All model identifiers are now managed in one place
- **Consistent ID generation**: Uses the following naming schema:
  - `classname`: Based on model class (e.g., "PLSRegression")
  - `name`: Custom name if provided, otherwise classname
  - `model_id`: Unique ID for the run (name + operation_counter)
  - `model_uuid`: Global unique ID (model_id + fold + step + config)
- **Caching**: Maintains a cache of identifiers for consistency across the pipeline

### 2. New CVAveragingManager (`cv_averaging.py`)

- **Automatic average prediction generation**: After CV training, generates both avg and w-avg predictions
- **Uses existing prediction helpers**: Leverages `PredictionHelpers.calculate_average_predictions()` and `calculate_weighted_average_predictions()`
- **Proper integration**: Works seamlessly with the existing prediction storage system

### 3. Updated AbstractModelController

#### New Components Integration:
```python
def __init__(self):
    # ... existing components ...
    self.naming_manager = ModelNamingManager()
    self.averaging_manager = CVAveragingManager(self.naming_manager)
```

#### Enhanced CV Execution:
- **Separation of concerns**: Clear distinction between finetuning and training phases
- **Automatic avg/w-avg generation**: After fold training completes, average predictions are generated
- **Consistent naming**: All model operations use the centralized naming manager

#### Improved Prediction Storage:
- **Consistent model UUIDs**: Uses naming manager for all prediction storage
- **Better organization**: Clear separation between fold predictions and averaged predictions

### 4. Updated CV Strategies

#### SimpleCVStrategy:
- **Two-phase approach**:
  1. Phase 1: Finetuning on combined data (if finetuning enabled)
  2. Phase 2: Training on individual folds with optimized parameters
- **Parameter propagation**: Best parameters from finetuning are applied to fold models
- **Proper binary naming**: Fold binaries include fold information in names

#### GlobalAverageCVStrategy:
- **Global optimization**: Parameters optimized across all folds simultaneously
- **Consistent parameter application**: Best parameters applied to all fold models
- **Clean separation**: Finetuning and training phases are clearly separated

## Benefits

### 1. Model Naming Consistency
- **No more local reconstruction**: Model names are generated once and cached
- **Traceable IDs**: Every model has a unique, traceable identifier
- **Debugging improvements**: Clear naming makes debugging much easier

### 2. Proper CV Implementation
- **Clear separation**: Train and finetune are properly separated
- **Automatic averaging**: avg and w-avg predictions are generated automatically
- **Better organization**: Fold models are clearly distinguished from averaged models

### 3. Prediction Storage Improvements
- **Consistent schema**: All predictions use the same naming convention
- **Better metadata**: Enhanced metadata for easier analysis
- **Proper fold handling**: Fold predictions are stored with correct identifiers

### 4. Enhanced Maintainability
- **Modular design**: Each component has a single responsibility
- **Easy testing**: Components can be tested independently
- **Future extensibility**: Easy to add new CV strategies or naming schemes

## Usage Examples

### Model Naming
```python
# In any controller
identifiers = self.naming_manager.create_model_identifiers(
    model_config, runner, fold_idx=2
)
print(identifiers.model_uuid)  # "PLSRegression_15_fold2_step3_config_pipeline_Q1"
```

### Average Prediction Generation
```python
# Automatically called after CV training
avg_binaries, metadata = self.averaging_manager.generate_average_predictions(
    dataset=dataset,
    model_config=model_config,
    runner=runner,
    context=context,
    fold_count=3
)
```

## Migration Impact

### Existing Code Compatibility
- **External interfaces preserved**: All public methods maintain the same signatures
- **Backward compatibility**: Existing pipelines will continue to work
- **Gradual adoption**: New features can be adopted incrementally

### Performance Improvements
- **Reduced redundancy**: No more duplicate name generation
- **Better caching**: Model identifiers are cached for reuse
- **Cleaner execution**: Clearer separation of concerns improves performance

## Testing Recommendations

1. **Unit tests**: Test each new component independently
2. **Integration tests**: Test the full CV pipeline with the new components
3. **Regression tests**: Ensure existing functionality is preserved
4. **Performance tests**: Verify performance improvements

## Future Enhancements

1. **Additional CV strategies**: Easy to add new strategies with the existing framework
2. **Enhanced averaging**: Support for custom averaging strategies
3. **Better error handling**: More specific error handling for different failure modes
4. **Advanced caching**: More sophisticated caching strategies for large datasets

This refactoring provides a solid foundation for reliable, maintainable model training and prediction generation in the nirs4all framework.