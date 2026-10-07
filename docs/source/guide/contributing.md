# 12. Add an operation or contribute an example

**This chapter is optional.** You can build advanced pipelines using the existing nodes. Read it when your method, input format or example is missing and you want to extend the project.

**Your goal:** identify where the change belongs, make its behavior testable, and document it so another user can reproduce the same result.

A language binding should call the numerical implementation rather than recalculate the same method. An image encoder, for example, needs one tested fitting/prediction implementation plus the public language entry points that can execute it.

| Repository | Responsibility | Extension examples |
|---|---|---|
| DAG-ML | Graphs, phases, folds/OOF, lineage, scoring and result persistence | New native coordination contract or replay validation |
| DAG-ML-Data | Typed representations and aligned relations | Masks, source identity and data-view contracts |
| Formats | Vendor readers | New instrument payload parser and licensed fixtures |
| IO | Dataset assembly | Joins, source declarations and public dataset transport |
| Methods | Numerical estimators and portable learned state | Encoder, model, artifact version and C ABI binding |
| Core | Product facades and artifact containers | Thin tasks, runtime loaders, cross-language surface and packaging |
| Python SDK / R product | Idiomatic public workflows and host controllers | New operator/controller or host integration |
| Studio/Web/UI | Application orchestration and presentation | User flow and visualization of existing scientific results |

## Add a feature end to end

Define input/output, error and artifact contracts first. Implement the owned computation upstream. Add thin language bindings and a public task entry only where it can execute. Qualify the actual path with independent numerical references, identity/leakage checks, cold replay and negative/tamper cases. Document accepted and refused profiles, not merely exported symbols.

The current flow is configuration → validated dataset/graph → native plan → FIT_CV/SELECT/REFIT → persisted result/model → replay. The historical direct-Python controller architecture is an explicit legacy lane; it does not describe every default native execution path.

Use existing schema/registry exports to generate inventories. Adding a keyword to a descriptive registry does not itself implement parser or runtime support. Check the executable parser, controller and artifact consumer separately. Every nested option needs types, defaults, interactions and a minimal/composed example.

## Tests and releases

Run the changed repository's gates and cross-repo contract checks. Keep fixtures frozen unless their provenance justifies an update. Test installed artifacts after build/publication; source-tree success is not installed-package evidence. Record exact SHAs, versions, runtime/ABI, commands, pass/skip counts and declared external preconditions. Do not relabel a repaired subset as a fresh complete green invocation.

See {doc}`/developer/architecture`, {doc}`/developer/pipeline_architecture`, {doc}`/developer/controllers`, {doc}`/developer/testing`, {doc}`/developer/artifacts` and {doc}`/developer/documentation_style`. Native C ABI changes require explicit upstream versioning and resource-lifetime tests; no binding should bypass the public ABI.


## Follow an extension through the system


Explanation: a documented feature starts with a scientific/runtime
contract, is implemented in its owning component, gains thin host bindings and
is qualified through the installed consumer. An example should run through the actual public entry point and reload its saved model.

For a new supervised selector, specify whether `y` is required, which rows may
be observed during FIT_CV, how selected indices are persisted, and how PREDICT
validates the incoming feature order. A selector that computes indices on all
rows before CV leaks validation targets. Its replay must import those indices
rather than selecting variables again on the deployment cohort.

For a new multimodal encoder, describe representation/axes, missing-source
behavior, output dimension, fitted-state profile, and sample identity
preservation. For a new host estimator, declare how an installed consumer
rehydrates weights and which optional dependency versions it needs. Keep these
contracts close to the owned implementation, then link their public reference
from the common guide.

## Write documentation that a user can execute

A feature page should answer the following in order:

1. What scientific problem does it solve, and what does it assume?
2. What enters and leaves the node, including shapes, units, masks and IDs?
3. Which parameters are required, which have defaults, and which interact?
4. What is fitted on each training fold, what happens on validation, and what is
   saved for replay?
5. What is the smallest runnable pipeline, and how does it compose with other
   nodes without changing the evaluation protocol?
6. Which products execute it, which consumer replays it, and which profiles are
   refused or not qualified?
7. Which existing example/test supplies evidence, and what should the reader
   inspect in its output?

Provide the actual host API when it exists. A Python object shown inside an R
or WASM code fence does not document an interlanguage feature. JSON/YAML should
use executable parser keys; identify sketches explicitly when they are only
conceptual. Link exact examples instead of copying an entire script whose data
or output directory assumptions differ from the page.

## Figures and scientific evidence

Every figure needs a stated question, visible conclusion and text alternative.
Flow diagrams should describe all essential dependencies in a nearby paragraph.
For a numeric chart, include units, cohort/source, evaluation split and a table
or downloadable data when exact values matter. Label synthetic teaching data
as synthetic; do not present a smooth toy fit as evidence of instrument accuracy.

A preprocessing comparison should show both the transformed signal and a
held-out score under the same folds. A fusion comparison should state source
availability and whether meta-model inputs are out-of-fold. An uncertainty
chart should identify calibration/test cohorts and empirical coverage.
Reproduce figures with a checked-in script and fixed seed where sampling is
used. Accessible descriptions are part of the documentation contract.

## A practical review protocol

Check a minimal example first, then a composed case using the same feature.
Confirm parameter errors fail before fitting, masks/IDs survive each host
boundary, and replay works without fitting in a fresh process. Read back the
export with its actual consumer. Record environmental skips accurately.

Build the documentation after editing:

::::{tab-set}
:sync-group: language

:::{tab-item} JSON
:sync: json

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} YAML
:sync: yaml

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Python
:sync: python



```bash
sphinx-build -b html docs/source docs/_build/html --keep-going
```

:::

:::{tab-item} R
:sync: r

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} Octave / MATLAB
:sync: matlab

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

:::{tab-item} WASM / JavaScript
:sync: javascript

This example uses Python SDK objects. An equivalent public operation is unavailable in this host. Use the shared native recipe in {doc}`languages` for the executable cross-language path.

:::

::::

Inspect tabs, figure captions, links and code indentation in the rendered page.
When adding a new page, preserve the learning path and connect it to both the
common guide and the deeper reference. Prefer one authoritative explanation
with targeted links over diverging descriptions of the same API.
