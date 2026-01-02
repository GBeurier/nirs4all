# Examples

NIRS4ALL includes a comprehensive collection of examples organized into three categories. Each section provides a detailed walkthrough with explanations, code samples, and links to the full source code.

```{toctree}
:maxdepth: 2
:hidden:

user/getting_started
user/data_handling
user/preprocessing
user/models
user/cross_validation
user/deployment
user/explainability
developer
```

## User Examples

Step-by-step examples for common NIRS workflows, organized by topic. Start here if you're new to NIRS4ALL.

| Section | Description | Examples |
|---------|-------------|----------|
| {doc}`user/getting_started` | Your first pipelines | Hello world, regression, classification, visualization |
| {doc}`user/data_handling` | Data input and formats | Flexible inputs, multi-datasets, multi-source, wavelengths, synthetic |
| {doc}`user/preprocessing` | NIRS transformations | SNV, MSC, derivatives, smoothing, augmentation |
| {doc}`user/models` | Model training | Multi-model, tuning, stacking, PLS variants |
| {doc}`user/cross_validation` | Validation strategies | KFold, stratified, group splitting, aggregation |
| {doc}`user/deployment` | Production deployment | Save/load, export bundles, sklearn integration |
| {doc}`user/explainability` | Model interpretation | SHAP basics, feature selection |

## Developer Examples

Advanced examples for extending NIRS4ALL capabilities.

| Section | Description |
|---------|-------------|
| {doc}`developer` | Branching, generators, deep learning, transfer learning, custom controllers |

## Quick Start

### Running Examples

```bash
cd examples

# Run all examples
./run.sh

# Run only user examples
./run.sh -c user

# Run by name pattern
./run.sh -n "U01*.py"

# Run with plots
./run.sh -p -s

# Quick mode (skip deep learning)
./run.sh -q
```

### Running Directly

```bash
# From project root
python examples/user/01_getting_started/U01_hello_world.py

# With visualization
python examples/user/01_getting_started/U02_basic_regression.py --plots --show
```

## Example Structure

```text
examples/
├── user/                    # User-facing examples
│   ├── 01_getting_started/  # U01-U04: First steps
│   ├── 02_data_handling/    # U01-U06: Data formats
│   ├── 03_preprocessing/    # U01-U04: NIRS transforms
│   ├── 04_models/           # U01-U04: Model training
│   ├── 05_cross_validation/ # U01-U04: CV strategies
│   ├── 06_deployment/       # U01-U04: Production
│   └── 07_explainability/   # U01-U03: SHAP
├── developer/               # Advanced developer examples
│   ├── 01_advanced_pipelines/
│   ├── 02_generators/
│   ├── 03_deep_learning/
│   ├── 04_transfer_learning/
│   ├── 05_advanced_features/
│   └── 06_internals/
├── reference/               # Reference examples (R01-R04)
└── sample_data/             # Sample datasets
```

## Learning Path

### Beginner Path

1. **Start Here**: {doc}`user/getting_started` - Learn the basics
2. **Data Loading**: {doc}`user/data_handling` - Understand input formats
3. **Preprocessing**: {doc}`user/preprocessing` - NIRS-specific transforms
4. **Models**: {doc}`user/models` - Train and compare models

### Intermediate Path

5. **Validation**: {doc}`user/cross_validation` - Proper evaluation
6. **Deployment**: {doc}`user/deployment` - Save and deploy models
7. **Explainability**: {doc}`user/explainability` - Understand predictions

### Advanced Path

8. **Developer**: {doc}`developer` - Extend NIRS4ALL

## Quick Links

- {doc}`/getting_started/quickstart` - Getting started guide
- {doc}`/user_guide/index` - Complete user guide
- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference
