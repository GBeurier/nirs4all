# Learn to build, understand and reuse pipelines

**Start with one complete experiment. End with several sources, branches, a model search and a saved predictor.** This course is for people who know their measurements and want to understand every step between a raw observation and a prediction. Follow it from Python, R, Octave or a browser.

Our question is **“Can these measurements predict the concentration of a sample?”** A pipeline is the recipe: prepare inputs, train a model, check predictions on observations it has not seen, and save the selected model. Images, time series and laboratory metadata can join the spectrum later.

```{figure} /assets/guide/workflow.svg
:alt: Data are split for candidate comparison; the selected recipe is fitted, saved, and used on new observations.

**The complete journey.** Validation answers which recipe to keep. The final fit creates the predictor you reuse. Future observations use its learned transformations and coefficients.
```

## Choose your starting point

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 1 · Run a first experiment
:link: start
:link-type: doc

Install your tools, fit a small model and save a result you can reload.
:::

:::{grid-item-card} 2 · Understand the data
:link: datasets
:link-type: doc

Read rows, columns, IDs, groups and multiple sources. Align them correctly.
:::

:::{grid-item-card} 3 · Explore every pipeline node
:link: /reference/nodes/index
:link-type: doc

A clickable catalogue: purpose, input/output, example and illustration.
:::

:::{grid-item-card} 4 · Build and compare recipes
:link: pipelines
:link-type: doc

Generate candidates, branch the data, merge features and build stacking.
:::

:::{grid-item-card} 5 · Read the evidence
:link: evaluation
:link-type: doc

Know which score answers your question and what your figures should reveal.
:::

:::{grid-item-card} 6 · Predict in your language
:link: languages
:link-type: doc

Use a shared recipe in Python, R, Octave and WASM; export and reload.
:::
::::

## Follow the course in order

| Chapter | You will learn to… | Check before continuing |
|---|---|---|
| {doc}`interfaces` | Choose your language's installed product | Identify the package you actually call |
| {doc}`start` | Train and save a predictor | Find two candidates and a saved model |
| {doc}`principles` | Explain learning, validation and refit | Distinguish settings from learned coefficients |
| {doc}`tasks` | Choose run, generate, predict or retrain | State what you have and what you need |
| {doc}`datasets` | Describe one or several measurement sources | Match IDs and explain one row |
| {doc}`pipelines` | Combine nodes and compare alternatives | Draw the data reaching each model |
| {doc}`evaluation` | Choose folds and interpret scores | Keep a specimen on one side of a split |
| {doc}`results` | Make a scientific report | Identify the winner and its evaluation data |
| {doc}`deployment` | Reload the complete predictor | Predict without fitting again |
| {doc}`catalog` | Choose preprocessing and models | Explain the expected signal change |
| {doc}`languages` | Run the shared example in your language | Compare values and row identities |
| {doc}`tutorials` | Assemble an advanced workflow | Complete branching and source-fusion exercises |
| {doc}`contributing` | Add an operation when needed | Locate numerical code and its tests |
| {doc}`reference` | Look up exact keywords and parameters | Reach the relevant example directly |

## How to read the examples

Each code box groups **all applicable languages** in synchronized tabs: JSON/YAML recipes and Python, R, Octave/MATLAB or WASM/JavaScript execution where supported. Your selection follows you between chapters. Runtime limitations and executable alternatives are explained beside the examples; installation tabs contain only actual language tools.

The shared recipe uses native method IDs. Advanced Python recipes use sklearn-compatible objects. They are two supported ways to work with different operator coverage; {doc}`interfaces` explains the choice.

Figures explain changes to data and expected results. Synthetic examples make arithmetic inspectable; their curves and scores are teaching examples, not claims about your instrument.

```{toctree}
:maxdepth: 2
:hidden:

interfaces
start
principles
tasks
datasets
pipelines
evaluation
results
deployment
catalog
languages
tutorials
contributing
reference
```
