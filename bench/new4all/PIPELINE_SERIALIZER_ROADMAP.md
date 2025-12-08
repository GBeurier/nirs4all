# Pipeline Serializer Roadmap

## Overview

A comprehensive pipeline serialization system that enables complete reproducibility and reusability of NIRS ML pipelines. The serializer handles four main components:

1. **Pipeline Configuration** (YAML/JSON)
2. **Fitted Objects** (ZIP/PKL bundles)
3. **Trained Models** (Framework-specific formats)
4. **Dataset Folds** (SpectraDataset serialization)

## Architecture Design

```
PipelineSerializer
├── ConfigSerializer      # YAML/JSON pipeline configs
├── ObjectSerializer      # Fitted transformers, clusterers
├── ModelSerializer       # .keras, .pkl, .ckpt models
└── DatasetSerializer     # SpectraDataset folds & splits
```

## Phase 1: Core Serialization Infrastructure

### 1.1 PipelineSerializer Class
```python
class PipelineSerializer:
    """Comprehensive pipeline serialization system"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.config_serializer = ConfigSerializer()
        self.object_serializer = ObjectSerializer()
        self.model_serializer = ModelSerializer()
        self.dataset_serializer = DatasetSerializer()

    def save_complete_pipeline(self, pipeline_data: Dict, dataset: SpectraDataset):
        """Save all pipeline components"""

    def load_complete_pipeline(self, bundle_path: str) -> Dict:
        """Load all pipeline components for reuse"""
```

### 1.2 ConfigSerializer
```python
class ConfigSerializer:
    """Handle pipeline configuration serialization"""

    def save_config(self, config: Dict, format: str = 'yaml') -> str
    def load_config(self, filepath: str) -> Dict
    def validate_config(self, config: Dict) -> bool
    def convert_format(self, input_path: str, output_format: str) -> str
```

### 1.3 ObjectSerializer
```python
class ObjectSerializer:
    """Handle fitted scikit-learn objects and transformers"""

    def save_fitted_objects(self, objects: Dict[str, Any], bundle_path: str)
    def load_fitted_objects(self, bundle_path: str) -> Dict[str, Any]
    def create_object_manifest(self, objects: Dict) -> Dict
    def verify_object_integrity(self, bundle_path: str) -> bool
```

## Phase 2: Model Serialization

### 2.1 ModelSerializer
```python
class ModelSerializer:
    """Handle different ML framework model formats"""

    # Framework-specific handlers
    def save_sklearn_model(self, model, filepath: str)
    def save_keras_model(self, model, filepath: str)
    def save_pytorch_model(self, model, filepath: str)
    def save_xgboost_model(self, model, filepath: str)

    # Generic interface
    def save_model(self, model, filepath: str, format: str = 'auto')
    def load_model(self, filepath: str, framework: str = 'auto')
    def get_model_info(self, filepath: str) -> Dict
```

### 2.2 Model Format Support

| Framework | Save Format | Load Support | Metadata |
|-----------|-------------|--------------|----------|
| scikit-learn | `.pkl` | ✅ | Full |
| Keras/TensorFlow | `.keras` | ✅ | Full |
| PyTorch | `.pth` | ✅ | Full |
| XGBoost | `.pkl`, `.json` | ✅ | Full |
| LightGBM | `.pkl`, `.txt` | ✅ | Partial |

## Phase 3: Dataset Serialization

### 3.1 DatasetSerializer
```python
class DatasetSerializer:
    """Handle SpectraDataset serialization with fold preservation"""

    def save_dataset_folds(self, dataset: SpectraDataset, bundle_path: str)
    def load_dataset_folds(self, bundle_path: str) -> SpectraDataset
    def save_split_info(self, dataset: SpectraDataset) -> Dict
    def recreate_splits(self, dataset: SpectraDataset, split_info: Dict)
```

### 3.2 SpectraDataset Extensions Needed

```python
# Add to SpectraDataset class:

def get_split_info(self) -> Dict:
    """Extract current train/test/val splits"""
    return {
        'train_indices': self.get_partition_indices('train'),
        'test_indices': self.get_partition_indices('test'),
        'val_indices': self.get_partition_indices('val'),
        'split_strategy': self.split_metadata,
        'random_state': self.random_state
    }

def apply_split_info(self, split_info: Dict):
    """Recreate exact train/test/val splits"""

def serialize_to_dict(self) -> Dict:
    """Full dataset serialization"""

def deserialize_from_dict(self, data: Dict):
    """Full dataset deserialization"""
```

## Phase 4: Bundle Management

### 4.1 Bundle Structure
```
pipeline_bundle.zip
├── metadata/
│   ├── manifest.json          # Bundle contents & versions
│   ├── pipeline_info.json     # Execution history, performance
│   └── environment.json       # Dependencies, versions
├── config/
│   ├── pipeline_config.yaml   # Original pipeline configuration
│   └── pipeline_config.json   # JSON equivalent
├── objects/
│   ├── fitted_transformers.pkl # Fitted preprocessing steps
│   ├── fitted_clusterers.pkl   # Fitted clustering operations
│   └── object_manifest.json    # Object metadata & hashes
├── models/
│   ├── model_1.keras          # Trained neural networks
│   ├── model_2.pkl            # Trained sklearn models
│   └── model_manifest.json    # Model metadata
└── data/
    ├── dataset_folds.pkl      # SpectraDataset with splits
    ├── split_info.json        # Split configuration
    └── feature_info.json      # Feature engineering steps
```

### 4.2 Bundle Versioning
```python
class BundleVersionManager:
    """Handle bundle versioning and compatibility"""

    def create_version_info(self) -> Dict
    def check_compatibility(self, bundle_path: str) -> bool
    def upgrade_bundle(self, old_bundle: str, new_version: str) -> str
    def compare_bundles(self, bundle1: str, bundle2: str) -> Dict
```

## Phase 5: Integration with PipelineRunner

### 5.1 Enhanced Runner Methods
```python
class PipelineRunner:
    # ...existing methods...

    def save_complete_pipeline(self,
                              filepath: str,
                              dataset: SpectraDataset,
                              include_data: bool = True,
                              compression_level: int = 6) -> str:
        """Save complete pipeline with all artifacts"""

    def load_pipeline_for_prediction(self, bundle_path: str) -> 'PipelinePredictors':
        """Load fitted pipeline for prediction on new data"""

    def load_pipeline_for_retraining(self, bundle_path: str) -> Tuple[Dict, SpectraDataset]:
        """Load pipeline for retraining or modification"""
```

### 5.2 Pipeline Predictors
```python
class PipelinePredictors:
    """Loaded pipeline optimized for prediction"""

    def __init__(self, bundle_path: str):
        self.fitted_objects = self._load_fitted_objects(bundle_path)
        self.trained_models = self._load_trained_models(bundle_path)
        self.preprocessing_chain = self._build_preprocessing_chain()

    def predict(self, new_data: SpectraDataset) -> np.ndarray:
        """Apply full pipeline to new data"""

    def predict_proba(self, new_data: SpectraDataset) -> np.ndarray:
        """Get prediction probabilities"""

    def transform_only(self, new_data: SpectraDataset) -> SpectraDataset:
        """Apply only preprocessing steps"""
```

## Implementation Timeline

### Week 1-2: Core Infrastructure
- [ ] PipelineSerializer base class
- [ ] ConfigSerializer (YAML/JSON)
- [ ] Basic ObjectSerializer (pkl)
- [ ] Simple bundle creation

### Week 3-4: Model Support
- [ ] ModelSerializer framework detection
- [ ] scikit-learn model serialization
- [ ] Keras/TensorFlow model serialization
- [ ] Model metadata extraction

### Week 5-6: Dataset Integration
- [ ] SpectraDataset serialization extensions
- [ ] Split preservation logic
- [ ] Feature engineering tracking
- [ ] Data integrity verification

### Week 7-8: Bundle Management
- [ ] Complete bundle creation
- [ ] Bundle versioning system
- [ ] Compatibility checking
- [ ] Bundle comparison tools

### Week 9-10: Runner Integration
- [ ] Enhanced save/load methods
- [ ] PipelinePredictors class
- [ ] Prediction optimization
- [ ] Documentation and examples

## Usage Examples

### Saving a Complete Pipeline
```python
# After running a pipeline
dataset, history = runner.run_pipeline(config, dataset)

# Save everything for future use
bundle_path = runner.save_complete_pipeline(
    filepath='./pipelines/nirs_model_v1.zip',
    dataset=dataset,
    include_data=True
)
```

### Loading for Prediction
```python
# Load trained pipeline for new predictions
predictor = PipelinePredictors('./pipelines/nirs_model_v1.zip')

# Apply to new data
predictions = predictor.predict(new_dataset)
probabilities = predictor.predict_proba(new_dataset)
```

### Loading for Retraining
```python
# Load for modification or retraining
original_config, original_dataset = runner.load_pipeline_for_retraining(
    './pipelines/nirs_model_v1.zip'
)

# Modify config and retrain
modified_config = {**original_config, 'model': {'type': 'RandomForest'}}
new_dataset, new_history = runner.run_pipeline(modified_config, original_dataset)
```

## Benefits

1. **Complete Reproducibility:** Exact recreation of any pipeline run
2. **Easy Deployment:** Single bundle contains everything needed
3. **Version Control:** Track pipeline evolution and performance
4. **Collaboration:** Share complete pipelines between team members
5. **A/B Testing:** Compare different pipeline configurations
6. **Production Ready:** Optimized prediction interface for deployment

## Technical Considerations

### Memory Efficiency
- Lazy loading of large components
- Memory-mapped arrays for large datasets
- Compression options for different use cases

### Security
- Bundle integrity verification
- Secure pickle handling
- Dependency vulnerability scanning

### Performance
- Parallel loading of bundle components
- Caching of frequently used objects
- Optimized prediction pipelines

This roadmap provides a comprehensive plan for implementing a production-ready pipeline serialization system that addresses all the requirements for NIRS ML pipeline reproducibility and deployment.
