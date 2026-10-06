# 12. Develop and contribute

Extend the component that owns the behavior. A binding should delegate numerical and orchestration work rather than becoming a second implementation.

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
