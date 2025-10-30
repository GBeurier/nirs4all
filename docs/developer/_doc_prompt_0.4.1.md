Design of `PipelineRunner` refactoring to be more readable, faster, and easier to maintain, while keeping current behavior and better performances (without overengineering).

Context: I have a roadmap doc, but the underlying logic is not right. I will provide it only to show the already identified needs. `PipelineRunner` is the core class and must become cleaner and more stable. Today it accepts `DatasetConfig` or datasets, and `PipelineConfig` or concrete pipeline instances. It can execute a large set of pipelines in one shot (instantiation, serialization, execution, formatting, etc.) by generating pipelines from `PipelineConfig` and by handling multiple datasets, then aggregates results for later analysis.

Key problems:

* Orchestration of many pipelines and the logic of a single pipeline should be separated.
* I want both levels to expose ML-standard APIs (`fit`, `transform`, `predict`, etc.) so they can act like transformers/evaluators/models.
* I cannot use Optuna on a single pipeline or on a group of pipelines with aggregated objectives (best, mean, median, etc.).

Goals:

* Make the class clear, maintainable, documented (Google-style docstrings), and modular (`pipeline_components/`).
* Anticipate alignment with common ML/DL practices without breaking existing code now.

A run should accept:

* One or many of: `SpectroDataset`, `DatasetConfig`, NumPy arrays `(X[, y])`, pandas DataFrames, or CSV paths.
* One or many of: `Pipeline`, `PipelineConfig`, serialized pipeline files, or existing pipeline instances.

Near-term desiderata to anticipate:

* Save a pipeline as a single file and reload it to run on another dataset with one command (predict, retrain, transfer, explain).
* Allow params at meta-pipeline and sub-pipeline levels; use Optuna to explore preprocessing, models, and hyperparameters.
* Make predictions a first-class object that a step can consume, enabling transparent stacking.

Task:

1. Read the provided code and docs.
2. Produce a single document: `pipeline_refactoring_logic_proposal.md`.

   * Ask questions where choices are unclear.
   * Offer alternatives when relevant.
   * Propose a management/architecture logic that improves maintainability and performance, and that anticipates treating both the global runner and sub-pipelines as standard ML pipeline objects, without immediate breaking changes (you can suggest future signature names to deprecate/introduce).
3. After we discuss and validate this proposal, write a detailed roadmap split into multiple documents (one per major phase).
4. We will then implement the roadmap.



#################


The files in utils are for most of them not at the right place.
evaluator, binning, balancing, task_type and task_detection, serialization should all be moved in a folder where it's make more sense.

Look at those files, look at the workplace, then move them, rename them, do what is necessary to have them in the right place, with the right intent and responsibility