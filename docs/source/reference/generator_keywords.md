# Generator keywords: choices, combinations and their exact outputs

Start with {doc}`/reference/nodes/generators` for the six-recipe walkthrough.
This page answers the next question: **what exactly does each keyword produce?**
Each small example has the same input in JSON, YAML and Python. The Python tab
calls the real generator and prints the expanded values without fitting a model.
The letter examples are expansion exercises, not importable operators.

:::{note}
These tabs express the same **Python SDK workflow**. JSON and YAML are recipe
files; Python can also use sklearn/nirs4all objects. R, Octave and WASM native
pipeline facades do not execute this host-controller node directly.
For a recipe that runs in those languages, use {doc}`/guide/languages`.
:::

```{figure} /assets/guide/generators.svg
:alt: Two preprocessing choices and three model-complexity choices produce six recipes.
:width: 100%

Independent choice counts multiply. Inspect expansion before committing to model fitting.
```


## Clickable keyword index

| Task | Keywords |
|---|---|
| Generate choices | {ref}`generator-keyword-or`, {ref}`generator-keyword-range`, {ref}`generator-keyword-log-range`, {ref}`generator-keyword-sample` |
| Combine parameters or stages | {ref}`generator-keyword-grid`, {ref}`generator-keyword-zip`, {ref}`generator-keyword-cartesian`, {ref}`generator-keyword-chain` |
| Select sets or sequences | {ref}`generator-keyword-pick`, {ref}`generator-keyword-arrange`, {ref}`generator-keyword-then-pick`, {ref}`generator-keyword-then-arrange` |
| Limit/reproduce selection | {ref}`generator-keyword-count`, {ref}`generator-keyword-seed`, {ref}`generator-keyword-weights` |
| Constrain choices | {ref}`generator-keyword-mutex`, {ref}`generator-keyword-requires`, {ref}`generator-keyword-exclude` |
| Reuse choices | {ref}`generator-keyword-preset` |
| Advanced/reserved names | {ref}`generator-keyword-tags`, {ref}`generator-keyword-metadata`, {ref}`generator-keyword-depends-on` |

## Alternatives and numeric values

(generator-keyword-or)=
### `_or_`

Choose one alternative. Use this for one preprocessing operator, model family or categorical setting per recipe.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "SNV",
    "MSC",
    "raw"
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- SNV
- MSC
- raw
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['SNV', 'MSC', 'raw']}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** `SNV`, `MSC`, `raw`: three independent choices.

At pipeline level, put actual importable operator descriptions in place of these labels. To apply several choices together, add `pick` or `arrange` below.

(generator-keyword-range)=
### `_range_`

Generate a numeric sequence with an **inclusive** upper bound. In `[start, end, step]`, the third value is the increment, not the number of points.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_range_": [
    2,
    8,
    2
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_range_:
- 2
- 8
- 2
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_range_': [2, 8, 2]}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** `[2, 4, 6, 8]`: four PLS component counts.

The two-value form `[2, 4]` defaults to step 1 and produces 2, 3, 4. A mapping can use `from`, `to`, `step`. Floating steps and descending ranges are supported. Step must be nonzero and point in the intended direction. Avoid assigning more PLS components than the training rank permits.

(generator-keyword-log-range)=
### `_log_range_`

Generate a specified **number of points** spaced multiplicatively. Use this when strengths span orders of magnitude.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_log_range_": [
    0.001,
    1.0,
    4
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_log_range_:
- 0.001
- 1.0
- 4
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_log_range_': [0.001, 1.0, 4]}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** approximately `[0.001, 0.01, 0.1, 1.0]`.

The third value is `num`, unlike `_range_` where it is a step. Endpoints must be positive. A mapping can use `from`, `to`, `num`; prefer explicit `num` to avoid ambiguity.

(generator-keyword-sample)=
### `_sample_`

Draw parameter values from a probability distribution. This is random-search planning, not sampling training observations.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_sample_": {
    "distribution": "uniform",
    "from": 0,
    "to": 1,
    "num": 3
  },
  "_seed_": 42
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_sample_:
  distribution: uniform
  from: 0
  to: 1
  num: 3
_seed_: 42
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_sample_': {'distribution': 'uniform', 'from': 0, 'to': 1, 'num': 3}, '_seed_': 42}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** approximately `[0.6394268, 0.0250108, 0.2750293]`.

`_seed_` makes the draw repeatable. Without a seed, repeated expansion may produce different candidate values.

| Distribution | Required settings | Useful for |
|---|---|---|
| `uniform` | `from`, `to`, `num` | Bounded continuous parameters |
| `log_uniform` | Positive `from`, `to`, `num` | Regularization/learning rates across orders of magnitude |
| `normal` / `gaussian` | `mean`, `std`, `num` | Values centered around a plausible mean |
| `choice` | `values`, `num` | Categories; draws can repeat |

## Products, paired parameters and ordered alternatives

(generator-keyword-grid)=
### `_grid_`

Try every parameter combination. Each result is a mapping of constructor parameters.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_grid_": {
    "alpha": [
      0.1,
      1.0
    ],
    "fit_intercept": [
      true,
      false
    ]
  }
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_grid_:
  alpha:
  - 0.1
  - 1.0
  fit_intercept:
  - true
  - false
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_grid_': {'alpha': [0.1, 1.0], 'fit_intercept': [True, False]}}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** four mappings: `(0.1, true)`, `(0.1, false)`, `(1.0, true)`, `(1.0, false)`.

Place this under a serialized estimator's `params`. The two lists each have length 2, so the product has 4 results. Nested generators such as `_range_` can supply parameter values.

(generator-keyword-zip)=
### `_zip_`

Pair values by their positions instead of trying every combination. This is useful for approved smoothing-window/polynomial settings.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_zip_": {
    "window_length": [
      7,
      11,
      15
    ],
    "polyorder": [
      2,
      3,
      3
    ]
  }
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_zip_:
  window_length:
  - 7
  - 11
  - 15
  polyorder:
  - 2
  - 3
  - 3
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_zip_': {'window_length': [7, 11, 15], 'polyorder': [2, 3, 3]}}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** three mappings: `(7, 2)`, `(11, 3)`, `(15, 3)`.

`_grid_` would create nine combinations. `_zip_` creates three. **Unequal lists stop at the shortest list**; the current implementation does not raise a length-mismatch error. Keep lengths equal and inspect output so longer-list candidates are not lost.

(generator-keyword-cartesian)=
### `_cartesian_`

Combine a choice at each ordered stage. Each result is an ordered list, suitable for a preprocessing sequence.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_cartesian_": [
    {
      "_or_": [
        "SNV",
        "MSC"
      ]
    },
    {
      "_or_": [
        "smooth",
        "derivative"
      ]
    }
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_cartesian_:
- _or_:
  - SNV
  - MSC
- _or_:
  - smooth
  - derivative
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_cartesian_': [{'_or_': ['SNV', 'MSC']}, {'_or_': ['smooth', 'derivative']}]}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** `[SNV, smooth]`, `[SNV, derivative]`, `[MSC, smooth]`, `[MSC, derivative]`.

The argument must be a **list of stages**, not a named dictionary. Each recipe keeps the stage order. `null` can represent a skipped stage where the containing workflow accepts it. Optional `pick`/`arrange` then select from the resulting complete stage sequences; that is a second selection level, not a request to select transforms within one stage.

(generator-keyword-chain)=
### `_chain_`

Enumerate alternatives in a deliberate order, such as a baseline, an improved recipe and a more complex recipe.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_chain_": [
    {
      "label": "baseline"
    },
    {
      "label": "normalized"
    },
    {
      "label": "normalized_and_smoothed"
    }
  ],
  "count": 2
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_chain_:
- label: baseline
- label: normalized
- label: normalized_and_smoothed
count: 2
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_chain_': [{'label': 'baseline'},
             {'label': 'normalized'},
             {'label': 'normalized_and_smoothed'}],
 'count': 2}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** the first two mappings: baseline, then normalized.

**This is not sequential transformation.** A `preprocessing` list applies successive operators; `_chain_` lists candidate alternatives. With `count` and no seed it retains the first candidates. An explicit `_seed_` or API `seed` makes that limited selection random instead. Leave the seed unset if order is the point of this node.

## Sets and sequences: why order changes the count

With three options A, B and C:

| Selection | Results | Count |
|---|---|---:|
| `pick: 2` | AB, AC, BC | 3 |
| `arrange: 2` | AB, AC, BA, BC, CA, CB | 6 |

Parallel feature collections often use `pick`. Sequential preprocessing often
uses `arrange`: normalize then differentiate can differ from differentiate then
normalize. These modifiers choose without repeating an item within a selection.

(generator-keyword-pick)=
### `pick`

Select an unordered subset. This treats A+B and B+A as the same selected set.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C"
  ],
  "pick": 2
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
pick: 2
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C'], 'pick': 2}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** `[[A, B], [A, C], [B, C]]`.

For `n` choices taken `k` at a time there are `n! / (k! (n-k)!)` sets. `pick: [1, 2]` means every size from 1 through 2, producing three singles and three pairs here. The selected output preserves a deterministic column order; keep that order when replaying a model.

(generator-keyword-arrange)=
### `arrange`

Select an ordered sequence. A then B and B then A are separate choices.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C"
  ],
  "arrange": 2
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
arrange: 2
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C'], 'arrange': 2}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** six sequences: AB, AC, BA, BC, CA, CB.

There are `n! / (n-k)!` sequences. `arrange: [1, 2]` allows sequence lengths one through two. Use actual operator declarations inside a `preprocessing` wrapper when these sequences should become transform chains.

(generator-keyword-then-pick)=
### `then_pick`

Select unordered groups **from results of the first selection**. This builds collections of already-generated sequences.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C"
  ],
  "arrange": 2,
  "then_pick": 2
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
arrange: 2
then_pick: 2
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C'], 'arrange': 2, 'then_pick': 2}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** 15 unordered pairs of the six ordered two-item sequences. The first is `[[A, B], [A, C]]`.

First `arrange: 2` produces 6 sequences. Then `then_pick: 2` produces 6 choose 2 = 15 pairs of sequences. The output is nested. Use it only when the containing node expects a collection of sequences; it is not a plain list of individual operators.

(generator-keyword-then-arrange)=
### `then_arrange`

Select ordered collections **from results of the first selection**.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C"
  ],
  "pick": 2,
  "then_arrange": 2
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
pick: 2
then_arrange: 2
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C'], 'pick': 2, 'then_arrange': 2}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** six ordered pairs of the three subsets. First: `[[A, B], [A, C]]`; reversing that pair is another result.

First `pick: 2` produces 3 subsets. Then `then_arrange: 2` produces 3 × 2 = 6 sequences of subsets. Nested output requires a compatible consuming node. Start with `pick`/`arrange` before using these second-order modifiers.

## Search size and reproducibility

(generator-keyword-count)=
### `count`

Limit the number of expanded alternatives at this node. For `_or_`, this is a sampled subset, not the first items.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C",
    "D",
    "E"
  ],
  "count": 2,
  "_seed_": 42
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
- D
- E
count: 2
_seed_: 42
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C', 'D', 'E'], 'count': 2, '_seed_': 42}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** two distinct choices, reproducibly selected with seed 42.

A positive count limits results; zero or negative count is treated as no limit by the strategies. Set a positive value deliberately. A local count limits only its node: two sampled transforms combined with three component counts still yield six recipes. `_chain_` has the special ordered behavior explained above.

(generator-keyword-seed)=
### `_seed_`

Make random generator selection repeatable without changing model or splitter randomness.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C",
    "D"
  ],
  "count": 2,
  "_seed_": 17
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
- D
count: 2
_seed_: 17
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C', 'D'], 'count': 2, '_seed_': 17}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** the same two choices every time this specification is expanded.

Generator seed, splitter `random_state` and model `random_state` are separate settings. Fix each source of randomness relevant to your experiment. The API can also supply `seed`; a node-local `_seed_` overrides it.

(generator-keyword-weights)=
### `_weights_`

Bias random alternative selection toward some choices. This changes sampling probability, not statistical model coefficients.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C",
    "D"
  ],
  "count": 2,
  "_weights_": [
    3,
    1,
    1,
    1
  ],
  "_seed_": 42
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
- D
count: 2
_weights_:
- 3
- 1
- 1
- 1
_seed_: 42
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C', 'D'], 'count': 2, '_weights_': [3, 1, 1, 1], '_seed_': 42}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** two sampled choices; A has a larger selection weight than B, C or D.

Provide one weight per alternative, with valid nonnegative weights and a positive total. The example does not guarantee that A appears. Weighted selection is a search-budget choice, not evidence that A is scientifically preferable. `_weights_` supports simple `_or_` + `count` sampling; combining it with `pick`, `arrange`, `then_pick` or `then_arrange` is rejected.

(phase-4-production-keywords)=
## Constraints: remove choices you do not want to test

(generator-keyword-mutex)=
### `_mutex_`

Prevent specified items from appearing together. Use this to remove incompatible parallel choices.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C",
    "D"
  ],
  "pick": 2,
  "_mutex_": [
    [
      "A",
      "B"
    ]
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
- D
pick: 2
_mutex_:
- - A
  - B
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C', 'D'], 'pick': 2, '_mutex_': [['A', 'B']]}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** five pairs: AC, AD, BC, BD, CD. AB is removed.

Constraints match the actual selected items. In a real operator specification, use the corresponding complete item descriptions rather than unrelated labels. Inspect filtered expansion before fitting.

(generator-keyword-requires)=
### `_requires_`

If the first item is selected, require the other listed items too. The dependency is one-directional.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C",
    "D"
  ],
  "pick": 2,
  "_requires_": [
    [
      "A",
      "C"
    ]
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
- D
pick: 2
_requires_:
- - A
  - C
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C', 'D'], 'pick': 2, '_requires_': [['A', 'C']]}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** four pairs: AC, BC, BD, CD. A without C is removed.

C does not require A in this example. This is selection compatibility, not execution order. Use `arrange` or an explicit sequential list if A must run after C.

(generator-keyword-exclude)=
### `_exclude_`

Remove specific selected combinations after expansion.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B",
    "C",
    "D"
  ],
  "pick": 2,
  "_exclude_": [
    [
      "A",
      "C"
    ],
    [
      "B",
      "D"
    ]
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
- C
- D
pick: 2
_exclude_:
- - A
  - C
- - B
  - D
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B', 'C', 'D'], 'pick': 2, '_exclude_': [['A', 'C'], ['B', 'D']]}
print(expand_spec(spec))
```
:::

::::

**Expanded result:** four pairs: AB, AD, BC, CD.

`_mutex_` expresses a general incompatibility; `_exclude_` records particular combinations to omit. These filters apply to selected combinations; validate that the intended items match your actual operator descriptions.

## Reuse a search specification

(generator-keyword-preset)=
### `_preset_`

A preset gives a reusable name to a specification. Register it in the Python
process, resolve the reference, then expand it. A recipe naming an unknown preset
is not self-contained: share the preset definition as well.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_preset_": "documented_components"
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_preset_: documented_components
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import (
    expand_spec, register_preset, resolve_presets_recursive, unregister_preset,
)

register_preset("documented_components", {"_range_": [2, 4]}, overwrite=True)
try:
    resolved = resolve_presets_recursive({"_preset_": "documented_components"})
    print(expand_spec(resolved))
finally:
    unregister_preset("documented_components")
```
:::

::::


**Expanded result:** `[2, 3, 4]`. Resolve before expansion. Circular references
are rejected. Built-in preset registration is explicit; do not assume a name
exists just because another notebook registered it.

## Advanced names with limited effects

(generator-keyword-tags)=
### `_tags_`

Tags can describe a generator specification or a registered preset. They do not
create sample tags: use {doc}`/reference/nodes/tag` for that.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B"
  ],
  "_tags_": [
    "scatter_search"
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
_tags_:
- scatter_search
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B'], '_tags_': ['scatter_search']}
print(expand_spec(spec))
```
:::

::::


**Expanded result:** `['A', 'B']`. The plain expansion API does not attach the tag
to returned scalar choices. Keep annotations with the original recipe when you
need them for reporting.

(generator-keyword-metadata)=
### `_metadata_`

Metadata describes the generator specification. It does not become biological
sample metadata and does not change estimator parameters.

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    "A",
    "B"
  ],
  "_metadata_": {
    "purpose": "baseline comparison"
  }
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- A
- B
_metadata_:
  purpose: baseline comparison
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': ['A', 'B'], '_metadata_': {'purpose': 'baseline comparison'}}
print(expand_spec(spec))
```
:::

::::


**Expanded result:** `['A', 'B']`, without metadata injected into each choice.
Keep the original specification with its annotations.

(generator-keyword-depends-on)=
### `_depends_on_`

This name is registered in keyword utilities, but the current expansion
strategies do **not implement conditional expansion through this field**.
It is not an executable recipe for parameter dependency.

To express a conditional model family, enumerate complete valid alternatives
with `_or_`. For example, Ridge has `alpha`, whereas a forest has
`n_estimators`; each valid alternative carries only its own parameters:

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

```json
{
  "_or_": [
    {
      "class": "sklearn.linear_model.Ridge",
      "params": {
        "alpha": 0.1
      }
    },
    {
      "class": "sklearn.ensemble.RandomForestRegressor",
      "params": {
        "n_estimators": 100,
        "random_state": 17
      }
    }
  ]
}
```
:::

:::{tab-item} YAML
:sync: yaml

```yaml
_or_:
- class: sklearn.linear_model.Ridge
  params:
    alpha: 0.1
- class: sklearn.ensemble.RandomForestRegressor
  params:
    n_estimators: 100
    random_state: 17
```
:::

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import expand_spec

spec = {'_or_': [{'class': 'sklearn.linear_model.Ridge', 'params': {'alpha': 0.1}},
          {'class': 'sklearn.ensemble.RandomForestRegressor',
           'params': {'n_estimators': 100, 'random_state': 17}}]}
print(expand_spec(spec))
```
:::

::::


**Expanded result:** two valid serialized estimator mappings. Wrap the selected
mapping under `model` in an actual pipeline.

## Inspect before fitting

Expansion is cheap relative to fitting, but enormous products can still consume
memory. `PipelineConfigs` also checks a generation ceiling (default 10,000).
Start by counting, expanding a small specimen, and reviewing the concrete recipes.

::::{tab-set}
:sync-group: language

:::{tab-item} Python
:sync: python

```python
from nirs4all.pipeline.config.generator import (
    count_combinations, expand_spec, expand_spec_iter, validate_spec,
)

spec = {"_grid_": {"alpha": [0.1, 1.0], "fit_intercept": [True, False]}}
validation = validate_spec(spec)
if not validation.is_valid:
    raise ValueError(validation.errors)
print("Candidate count:", count_combinations(spec))
for candidate in expand_spec(spec):
    print(candidate)
# For a large compatible specification, iterate without keeping every result:
for candidate in expand_spec_iter(spec):
    print(candidate)
```
:::

::::


**Expected result:** candidate count 4 and the four parameter mappings listed in
the `_grid_` example. Iteration changes storage behavior, not the experiment's
meaning. Constraints can filter candidates; inspect actual expanded results
rather than relying only on a theoretical count.

Generator tools also provide `expand_spec_with_choices` for choice tracking,
`batch_iter` for batches, `to_dataframe`/`format_config_table` for inspection and
preset export/import. Their Python API signatures are documented in
{doc}`/reference/combination_generator`.

Executable tutorials:
[D01 generator syntax](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D01_generator_syntax.py),
[D02 advanced generation](https://github.com/nirs4all/nirs4all/blob/main/examples/developer/02_generators/D02_generator_advanced.py).
