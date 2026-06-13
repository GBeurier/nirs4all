# Tutorial: Heterogeneous Source Repetitions

This tutorial walks through the experimental relation pipeline with the sample
fixtures shipped in `examples/sample_data/heterogeneous/`.

The example has five physical samples. Each sample is measured by three
sources, but each source has its own repetition count:

| Source | Repetitions per sample | Example files |
|--------|------------------------|---------------|
| MIR | 2 | `MIR_train.csv`, `MIR_test.csv` |
| RAMAN | 3 | `RAMAN_train.csv`, `RAMAN_test.csv` |
| NIRS | 2 | `NIRS_train.csv`, `NIRS_test.csv` |

This is not the same as legacy `repetition="sample_id"`, where all rows are
already in one rectangular table. Here, each source first needs an explicit
sample identity contract.

## 1. Start From A Relation Config

The simplest example is:

```text
examples/configs/datasets/heterogeneous_repetitions_per_source_aggregate.yaml
```

It declares the sample key, the source-specific repetition columns, the expected
cardinalities, and the first materialized representation:

```yaml
experimental_relation_pipeline: true

repetition_spec:
  sample_id: sample_id
  link_by: sample_id
  target_level: physical_sample
  rep_order: exchangeable
  strict_cardinality: true
  missing_repetition_policy: strict
  missing_source_policy: strict
  sources:
    MIR: {rep_col: rep, expected: 2}
    RAMAN: {rep_col: rep, expected: 3}
    NIRS: {rep_col: rep, expected: 2}

representations:
  - name: per_source_aggregate
    unit_level: sample
    method: mean
```

## 2. Parse The Contract

Use `parse_config` to load the YAML and `RepetitionSpec` to normalize the
relation contract:

```python
from pathlib import Path

from nirs4all.data.config_parser import parse_config
from nirs4all.data.relations import RepetitionSpec

examples_root = Path("examples")
config_path = examples_root / "configs/datasets/heterogeneous_repetitions_per_source_aggregate.yaml"

config, name = parse_config(str(config_path))
spec = RepetitionSpec.from_config(config["repetition_spec"])

assert config["experimental_relation_pipeline"] is True
assert name == "heterogeneous_repetitions_per_source_aggregate"
```

## 3. Build A Raw Multi-source Dataset

`RawMultiSourceDataset` is the staging object. It keeps the source tables
separate until you choose an explicit representation.

```python
import csv
import numpy as np

from nirs4all.data.raw_multisource import RawMultiSourceDataset


def read_target_map(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        return {row["sample_id"]: float(row["target"]) for row in csv.DictReader(handle)}


def read_source(path: Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    feature_headers = [column for column in rows[0] if column not in {"sample_id", "rep"}]
    X = np.asarray([[float(row[column]) for column in feature_headers] for row in rows])
    sample_ids = [row["sample_id"] for row in rows]
    rep_ids = [row["rep"] for row in rows]
    return X, sample_ids, rep_ids, feature_headers


# parse_config normalizes shared_targets to targets.
target_by_sample = read_target_map(examples_root / config["targets"]["path"])

X_by_source = {}
keys_by_source = {}
rep_by_source = {}
headers_by_source = {}
targets_by_source = {}

for index, source in enumerate(config["sources"]):
    source_id = source["name"]
    X, sample_ids, rep_ids, headers = read_source(examples_root / source["train_x"])
    X_by_source[source_id] = X
    keys_by_source[source_id] = sample_ids
    rep_by_source[source_id] = rep_ids
    headers_by_source[source_id] = headers

    # One source is enough to carry sample-level targets; they are linked by sample_id.
    if index == 0:
        targets_by_source[source_id] = [target_by_sample[sample_id] for sample_id in sample_ids]

raw = RawMultiSourceDataset.from_sources(
    spec,
    X_by_source,
    keys_by_source,
    headers_by_source=headers_by_source,
    rep_by_source=rep_by_source,
    targets_by_source=targets_by_source,
)
```

At this point, no rectangular model matrix has been created yet.

```python
print(raw.physical_sample_ids)
# ['S001', 'S002', 'S003']

print(raw.cardinalities()[("S001", "MIR")])
# 2

print(raw.cardinalities()[("S001", "RAMAN")])
# 3
```

## 4. Materialize A Sample-level Matrix

`per_source_aggregate` averages repetitions inside each source, then
concatenates the source blocks in canonical source order. The training fixture
has three physical samples and three wavelengths per source, so the resulting
matrix has shape `(3, 9)`.

```python
materialized = raw.materialize("per_source_aggregate")
model_X, model_headers = materialized.to_feature_matrix()

print(model_X.shape)
# (3, 9)

print(model_headers)
# ['MIR:w1000', 'MIR:w1100', 'MIR:w1200',
#  'NIRS:w1450', 'NIRS:w1550', 'NIRS:w1650',
#  'RAMAN:w500', 'RAMAN:w750', 'RAMAN:w1000']

print(materialized.sample_ids)
# ['S001', 'S002', 'S003']
```

This representation is the safest default when you want a conventional model:
one row per physical sample, no row-count bias from sources with more
repetitions.

## 5. Try A Cartesian Representation

When cross-source feature interactions matter, use a bounded cartesian plan.
For MIR=2, RAMAN=3, and NIRS=2, each sample has `2 * 3 * 2 = 12` combinations.

```python
from nirs4all.data.raw_multisource import RepresentationPlan

cartesian = raw.materialize(
    RepresentationPlan(
        "cartesian_full",
        max_combos_per_sample=12,
        max_total_combos=36,
        max_total_rows=36,
    )
)

print(cartesian.X.shape)
# (36, 9)

print(cartesian.unit_ids[0])
# S001::MIR1xNIRS1xRAMAN1
```

Always keep a cap on cartesian materialization. It makes the memory and replay
contract explicit, and it lets validation fail before allocating a matrix that
is larger than expected.

## 6. Use The Pipeline Boundary

Once a relation-aware dataset is staged, `rep_fusion` is the pipeline boundary
that converts it into a regular feature matrix for downstream operators:

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from nirs4all.pipeline.config import PipelineConfigs

pipeline = PipelineConfigs(
    [
        {"rep_fusion": "per_source_aggregate"},
        StandardScaler(),
        GroupKFold(n_splits=3),
        {"model": PLSRegression(n_components=2)},
    ]
)
```

In relation-aware runs, keep splitting and scoring at the physical-sample
level. The replay manifest stored with the run records the representation,
reducers, missingness policy, and fit-influence contract needed for bundle
prediction.

## 7. Compare The Shipped Examples

The repository includes four relation-contract examples:

| Config | Use when |
|--------|----------|
| `heterogeneous_repetitions_per_source_aggregate.yaml` | You want one conventional row per physical sample. |
| `heterogeneous_repetitions_late_fusion.yaml` | You want source-level branches and sample-aligned meta features. |
| `heterogeneous_repetitions_cartesian_full.yaml` | You need explicit cross-source combo rows with sample-level reducers. |
| `heterogeneous_repetitions_missing_source.yaml` | Prediction can miss a declared source and the model expects masks/padding. |

Mirrors are available under `examples/sample_configs/datasets/G01_...` through
`G04_...` for test-dataset discovery.

:::{note}
The Category G YAML files are experimental relation contracts. They are parsed
and covered by relation-table smoke tests, but they are not legacy
`DatasetConfigs` loader fixtures. Use them as contracts for source-aware
staging, materialization, and replay.
:::
