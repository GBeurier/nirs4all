# Cross-review v1 - ML_DATA + DAG-ML specifications

Date: 2026-05-22.
Scope: cross-read of 4 docs (`dag_ml_externalization_from_code.md`,
`ml_data_specification_v1.md`, `dag_ml_specification_v1.md`,
`dag_ml_use_cases.md`).
Method: types inventory + cross-references + algorithmic divergences +
UC coverage. Documents referenced as: **SRC** (externalization), **MD**
(ml_data spec), **DM** (dag_ml spec), **UC** (use cases).

---

## 1. Inventaire des types

Conventions: `Defined` = file:section / line. `Refs` = sites listing the
name as a type. `Status`: `OK` (single canonical def + coherent refs);
`DUPLICATE` (defined in 2+ docs, identical or near-identical);
`DIVERGENT` (defined in 2+ docs with differing fields);
`MISSING` (referenced but never formally defined);
`ORPHAN` (defined but never used elsewhere);
`AMBIGUOUS` (defined but signature/semantics conflict with usage).

### 1.1 Shared contract types (ML_DATA owns by §12 of MD)

| Type | Defined in | Referenced by | Status |
|---|---|---|---|
| `SampleId` (NewType str) | MD §2.1 / L118 | MD, DM §7.1 (`SampleIdT`), UC | `DIVERGENT` (DM redeclares as `SampleIdT = str` at DM:715 instead of importing) |
| `SourceId` | MD §2.1 / L119 | MD, DM, UC | OK |
| `RepresentationId` | MD §2.1 / L120 | MD, DM L126 (`PortSpec.representation`), UC | OK |
| `TypeId` | MD §2.1 / L121 | MD; DM does not explicitly reference (only via RepresentationId) | OK |
| `ObservationId` | MD §2.1 / L122 | MD §2.11; DM §8.1 uses `tuple[str, ...]` for `observation_ids` (untyped) | `AMBIGUOUS` (DM uses raw `str`) |
| `TargetId` | MD §2.1 / L123 | MD §2.11; DM §8.1 uses `tuple[str, ...]` for `target_ids` | `AMBIGUOUS` (DM uses raw `str`) |
| `GroupId` | MD §2.1 / L124 | MD §2.11; DM §7.2 uses `group_key: str` (a column name, not a `GroupId` value) | OK (different role) |
| `AxisKind` (Literal) | MD §2.2 / L145 | MD only | OK |
| `AxisSpec` | MD §2.2 / L153 | MD, DM §1.2 frontier list, UC indirectly | OK |
| `RepresentationSpec` | MD §2.3 / L174 | MD, DM (PortSpec.representation referenced by string), UC | OK |
| `SourceGranularity` (Literal) | MD §2.4 / L203 | MD, UC tables | OK |
| `SourceDescriptor` | MD §2.4 / L213 | MD, DM §1.2, UC tables | OK |
| `DatasetSchema` | MD §2.5 / L245 | MD, DM §5.1 PlanningContext, ExecutionBundle | OK |
| `DataView` | MD §2.6 / L266 | MD, DM (RunContext, NodeTask), UC | OK |
| `PresenceMask` | MD §2.7 / L291 | MD, UC (UC1 missing photo_side) | OK |
| `DataBlock` | MD §2.8 / L308 | MD, DM §1.2, UC | OK |
| `FeatureTable` | MD §2.9 / L340 | MD, DM §8.2 (oof_join return), UC | OK |
| `TargetBlock` | MD §2.10 / L362 | MD, DM §6.3 (ModelAdapter.fit), UC | OK |
| `SampleRelation` | MD §2.11 / L388 | MD, DM §7.3 §9.7, UC3, UC4, UC5 | OK |
| `AdaptationPolicy` | MD §5.1 / L592 | MD, DM §5.1 (PlanningPolicy.adaptation) | OK |
| `AdapterSpec` (ML_DATA) | MD §5.2 / L605 | MD; SRC §2.6.5 | `DIVERGENT` (see DM AdapterSpec below) |
| `AdapterContext` | MD §5.3 / L626 | MD; DM never imports despite usage | `AMBIGUOUS` (DM creates contexts but never names the type) |
| `FittedAdapter` | MD §5.4 / L641 | MD, DM §5.2 NodePlan, ExecutionBundle | OK |
| `RepresentationAdapter` (Protocol) | MD §5.5 / L651 | MD, DM §6.5 wraps it as ADAPTER NodeKind | OK |
| `AdapterRegistry` (Protocol) | MD §5.6 / L698 | MD, DM §9.1 (`ml_data.adapter_registry.get`) | OK (DM uses non-typed indirection) |
| `AlignmentPolicy` | MD §6.1 / L796 | MD, DM §5.1 PlanningPolicy.alignment, UC1 UC4 | OK |
| `FusionPolicy` | MD §6.2 / L822 | MD, DM §5.1, UC1 UC2 UC4 | OK |
| `AlignmentPlan` | MD §6.3 / L841 | MD only | ORPHAN cross-doc (used inside ML_DATA §6.4 only) |
| `FeatureJoiner` (Protocol) | MD §6.5 / L908 | MD, DM §9.4 (FeatureJoinNode wraps it) | OK |
| `InputPortSpec` | MD §7.1 / L993 | MD, DM §1.2 frontier list, never instantiated in DM | OK |
| `ModelInputSpec` | MD §7.1 / L1002 | MD, DM §6.2 DataAwareOperatorAdapter, UC | OK |
| `DataPlanStep` | MD §7.2 / L1014 | MD only | OK (internal to DataPlan) |
| `DataPlan` | MD §7.2 / L1022 | MD, DM §5.2 NodePlan, ExecutionBundle | OK |
| `DataPlanner` (Protocol) | MD §7.3 / L1036 | MD, DM §5.1 PlanningContext | `AMBIGUOUS` (signature mismatch, see §2.7) |
| `CollationPolicy` | MD §8.1 / L1273 | MD; DM never references | ORPHAN cross-doc |
| `BatchCollator` (Protocol) | MD §8.2 / L1285 | MD; DM §6.5 mentions `pytorch.module` uses it | OK |
| `AugmentationPlan` | MD §9.4 / L1392 | MD; DM §9.1 calls `adapter.plan(...)` but doesn't name the return | OK |
| `AugmentationAdapter` (Protocol) | MD §9.4 / L1397 | MD, DM §9.1, UC5 | OK |
| `AugmentationPolicy` | MD §9.4 / L1416 | MD §9.4; DM §9.1 / L980; UC5 | `DIVERGENT` (DM adds `multiplier` field; see §2.3) |
| `AuxInputSpec` | MD §10.1 / L1470 | MD, DM §2.3 NodeSpec.aux_inputs, UC1 | OK |
| `SerializableRef` | MD §11.3 / L1618 | MD, DM §2.3 SubgraphNodeSpec, ExecutionBundle | OK |
| `ArtifactSerializer` (Protocol) | MD §11.3 / L1624 | MD only | ORPHAN cross-doc |
| `MLDataset` (Protocol) | MD §3 / L432; redefined MD §10.2 / L1483 | MD, DM, UC | `DUPLICATE` within MD (second def adds `auxiliary()` method but is the same class signature) |
| `DataTypePlugin` (Protocol) | MD §4.2 / L508 | MD only | OK (registry-internal) |
| `DataTypeRegistry` (Protocol) | MD §4.3 / L537 | MD only | OK |
| `TypeCapability` | MD §4.1 / L497 | MD only | OK |

### 1.2 DAG-ML internal types (DM owns)

| Type | Defined in | Referenced by | Status |
|---|---|---|---|
| `NodeKind` (Enum) | DM §2.1 / L95 | DM, UC | `INCOMPLETE` (UC Annexe A adds `restructure`, `mixed_join`, `invariant_check`, `materialize`, `adapt`, `align`, `join`, `prediction`, `aggregator`, `sample_rel`, `search_space`, `tuner`, `subgraph`, `refit`, `restructure` as kinds; only a subset matches the enum) |
| `PortKind` (Literal) | DM §2.2 / L120 | DM, UC | OK |
| `PortSpec` | DM §2.2 / L123 | DM, UC10 | OK |
| `PortSchema` | DM §2.2 / L131 | DM | OK |
| `PortRef` | DM §2.2 / L136 | DM | OK |
| `EdgeContract` | DM §2.2 / L141 | DM | OK |
| `EdgeSpec` | DM §2.2 / L149 | DM | OK |
| `NodeSpec` | DM §2.3 / L159 | DM, UC | OK |
| `GraphInterface` | DM §2.3 / L171; also SRC §2.10.7 / L1355 | DM, UC10 | `DUPLICATE` (SRC is preliminary; DM canonical) |
| `GraphSpec` | DM §2.3 / L176; also SRC §2.10.7 / L1360 | DM, UC10, SRC | `DUPLICATE` (SRC + DM; both add fields; DM is canonical) |
| `SubgraphNodeSpec` | DM §2.3 / L185; also SRC §2.10.7 / L1368 | DM, UC10 | `DUPLICATE` (SRC has fewer fields; DM canonical) |
| `ResourceHints` | DM §2.4 / L209 | DM | OK |
| `Phase` (Enum) | DM §3.1 / L229 | DM, UC | OK |
| `SlotKind` (Literal) | DM §4.4 / L365 | DM | OK |
| `SlotSpec` | DM §4.4 / L370 | DM | OK |
| `SearchSpace` | DM §4.4 / L378 | DM, UC7 | OK |
| `PlanningContext` | DM §5.1 / L441 | DM | OK |
| `PlanningPolicy` | DM §5.1 / L450 | DM | OK |
| `NodePlan` | DM §5.2 / L464 | DM | OK |
| `ExecutionPlan` | DM §5.2 / L477 | DM, ExecutionBundle | OK |
| `PlanCacheStore` (Protocol) | DM §5.4 / L558 | DM only | OK |
| `OperatorAdapter` (Protocol) | DM §6.1 / L573; also SRC §1.4 / L100 | DM, SRC | `DUPLICATE` (SRC sketch, DM canonical) |
| `AdapterSpec` (DAG-ML) | DM §6.1 / L592 | DM | `DIVERGENT NAME` (collides with MD §5.2 `AdapterSpec`; DM explicitly comments "not the ML_DATA one") |
| `NodeTask` | DM §6.1 / L600 | DM | OK |
| `NodeResult` | DM §6.1 / L610 | DM | OK |
| `DataAwareOperatorAdapter` (Protocol) | DM §6.2 / L620 | DM | OK |
| `ModelAdapter` (Protocol) | DM §6.3 / L632 | DM | OK |
| `OperatorRegistry` (Protocol) | DM §6.4 / L663 | DM | OK |
| `Fold` | DM §7.1 / L718 | DM | OK |
| `FoldSet` | DM §7.1 / L726 | DM, UC | OK |
| `SplitPolicy` | DM §7.2 / L738; also SRC §2.10.1 / L1130 | DM, UC | `DUPLICATE` (SRC + DM; DM canonical) |
| `PredictionBlock` | DM §8.1 / L775; also SRC §2.10.1 / L1142 | DM, UC | `DUPLICATE` (SRC sketch; DM canonical) |
| `PredictionStore` (Protocol) | DM §8.1 / L794 | DM | OK |
| `PredictionPayload` | DM §8.1 / L809 | DM (ModelAdapter return) | OK |
| `AggregationPolicy` | DM §8.3 / L934; also SRC §2.10.1 / L1123 | DM, UC3 | `DIVERGENT` (SRC version differs: SRC has `level`, `method`, `custom_aggregator`, `keep_observation_predictions`; DM v1 has `use_proba`, `coverage`, `missing_value`, `duplicate_resolution`, `fold_mismatch`, `aggregation_level`, `method`, `outlier_threshold`). UC3 mixes both: uses `level=`, `method=`, `keep_observation_predictions=`, `exclude_outliers=` — none from DM's set |
| `AugmentationPolicy` | DM §9.1 / L980; MD §9.4 / L1416 | DM, MD, UC5 | `DIVERGENT` (DM adds `multiplier` field) |
| `ForkPolicy` | DM §9.2 / L1021 | DM, UC8 | OK |
| `MapPolicy` | DM §9.3 / L1047 | DM | OK |
| `FeatureJoinPolicy` | DM §9.4 / L1061 | DM | OK |
| `Variant` | DM §10.2 / L1135 | DM | OK |
| `TrialResult` | DM §10.4 / L1155 | DM, UC7 | OK |
| `TunerAdapter` (Protocol) | DM §10.4 / L1164 | DM, UC7 | OK |
| `TuningNodeSpec` | DM §10.5 / L1184 | DM, UC7 | OK |
| `RankingPolicy` | DM §11.1 / L1223 | DM | OK |
| `SelectedGraph` | DM §11.2 / L1236 | DM, ExecutionBundle | OK |
| `RefitPlan` | DM §11.3 / L1254 | DM only | `ORPHAN`-leaning (defined but never instantiated; `refit_*` strategies in §11.5 call `execute(...)` directly without building a `RefitPlan`) |
| `SeedContext` | DM §12.1 / L1369; also SRC §2.10.3 / L1202 | DM, MD §9.5 (referenced but not imported), UC | `DUPLICATE` (SRC sketch; DM canonical) |
| `ArtifactRef` | DM §13.1 / L1416 | DM, ExecutionBundle | OK |
| `ArtifactStore` (Protocol) | DM §13.1 / L1425 | DM | OK |
| `LineageRecord` | DM §13.2 / L1444 | DM | OK |
| `LineageRecorder` (Protocol) | DM §13.2 / L1462 | DM | OK |
| `CacheKey` | DM §13.3 / L1472 | DM | OK |
| `CacheStore` (Protocol) | DM §13.3 / L1480 | DM | OK |
| `RunContext` | DM §14.1 / L1524; also SRC §2.2 / L258 | DM | `DUPLICATE` (SRC sketch; DM canonical) |
| `ExecutionContext` | DM §14.1 / L1540 | DM | OK |
| `Scheduler` (Protocol) | DM §14.2 / L1552 | DM | OK |
| `ScheduledTask` | DM §14.2 / L1562 | DM | OK |
| `DagMLError` + subclasses | DM §14.6 / L1618-1646 | DM | OK |
| `ErrorPayload` | DM §14.6 / L1655 | DM | OK |
| `MetricsLogger` (Protocol) | DM §14.8 / L1681 | DM | OK |
| `SerializedDataPlan` | DM §15.1 / L1702 | DM, ExecutionBundle | OK |
| `ExecutionBundle` | DM §15.1 / L1709; also SRC §2.10.5 / L1286 | DM, SRC, UC9 | `DUPLICATE` (DM canonical; UC9 references `bundle.data_schema_fingerprint` field that does not exist on DM's class — see §2.6) |
| `ExplainHooks` (Protocol) | DM §15.4 / L1802 | DM | OK |
| `ExplainerAdapter` (Protocol) | DM §15.4 / L1806 | DM | OK |
| `ExplainResult` | DM §15.4 / L1816 | DM | OK |

### 1.3 Types referenced but never formally defined (`MISSING`)

| Type / construct | Site of reference | Comment |
|---|---|---|
| `FeatureBlock` | UC12 §12.4 / L1603, UC12 §12.7 / L1656; SRC §1.6 / L151, §2.2 / L274 | Only sketched in SRC; absent from MD and DM v1. UC12 uses it as a first-class output of "feature branches" distinct from `PredictionBlock`. Implementation needs a decision: either alias to `DataBlock(repr=tabular_numeric)` or introduce as a real type. |
| `MixedJoinNode` / `mixed_join` (NodeKind) | UC12 §12.4-12.7; UC Annexe A | Referenced as a node kind; absent from DM `NodeKind` enum. |
| `AggregatorNode` / `aggregator` (NodeKind) | UC3 §3.4 / L438, Annexe A L1741 implicitly; DM §17 UC4 / L2131 | DM describes it as "MODEL with degenerate fit" but no formal kind + the spec lacks a dedicated agg adapter. |
| `restructure` (NodeKind) | UC Annexe A L1749-L1750 (`rep_to_sources`, `rep_to_pp`) | Annexe A maps DSL keywords to `restructure` kind; DM `NodeKind` enum does not include it; DM §4.3 instead says "data layout directive; not a DAG node, resolved at PLAN via DataView.extra". Direct contradiction. |
| `sample_rel` / `invariant_check` (NodeKind) | UC4 §4.4 / L578, UC12 §12.4 / L1626 | UC tables reference them; DM has no such kinds. |
| `OOFLeakageError` | UC5 §5.5 / L714, UC11 §11.4 / L1436 + L1448, UC11 §11.8 / L1528, UC Annexe E / L1821 | DM error taxonomy (§14.6) defines `OOFUnsafeUsageError` but not `OOFLeakageError`. UC uses the latter name throughout. Naming inconsistency. |
| `LeakageError` (augmentation) | UC5 §5.5 / L714; UC Annexe E / L1822 | Not in DM taxonomy; should be `OOFError` or new subclass. |
| `oof_safe: bool` (on PredictionBlock) | UC11 §11.5 / L1489 | DM `PredictionBlock` (§8.1) does not declare this field. |
| `explicit_confirmed_leakage` (policy field) | UC11 §11.4-11.5 | DM has only `unsafe_use_train` (§8.5); UC11 invents a second confirmation flag. Spec must converge. |
| `block_kind: "feature" \| "prediction"` discriminator | UC12 §12.6 / L1643 | Required for `MixedJoinNode`; not present in DM/MD type system. |
| `branches_as_features` / `branches_as_predictions` (merge policy fields) | UC12 §12.3 / L1581-L1582 | DM `merge` lowering (§4.3) does not document an `"all"` merge mode with these fields. |
| `LeakageError` plain class | UC5 §5.5 | Distinct from OOFError taxonomy. |
| `CacheConfig` (with `step_cache_max_mb`) | DM §13.4 / L1511 | Field referenced but no `CacheConfig` class is defined in DM v1. nirs4all has it in `config/cache_config.py` but the DAG-ML extraction does not reify it. |
| `ModelNode.n_jobs_folds` | DM §14.4 / L1584, §18 / L2172 | Mentioned as a knob, but DM defines no `ModelNode` class — only `MODEL` `NodeKind` and `ModelAdapter`. No place to put this field. |
| `SampleIdT` | DM §7.1 / L715, §8.1 / L777 | Reinvented as `SampleIdT = str`. Should import MD's `SampleId` NewType. |
| `CVResult`, `RunResult`, `RefitResult` | DM §16 (public API) / L1871, L1887, L1922 | Used as return types in the public API but never defined as dataclasses anywhere in DM. |
| `tabular.pca` adapter id | UC12 §12.3 / L1571 | UC names an adapter not listed in MD §5.8 core adapters. |
| `Aggregator` lineage / `keep_observation_predictions` | UC3 §3.3 / L411 | UC uses `AggregationPolicy(level=..., keep_observation_predictions=True, exclude_outliers={...})` but DM `AggregationPolicy` (§8.3) does not have these fields. |
| `requires_user_choice: list[Decision]` structured form | MD §7.6 (own open question) | Discussed as future; current spec has free-text strings. |
| `data_schema_fingerprint` (ExecutionBundle attr) | DM §15.2 / L1737 | Code reads `bundle.data_schema_fingerprint` but the bundle class only stores `data_schema: DatasetSchema`. The fingerprint must be (re)computed from MD's `schema_fingerprint(...)`. |
| `view_for_subset`, `isolate_chain`, `build_chain_plan`, `re_plan`, `fold_set_final`, `view_for_subset` (helpers) | DM §11.5 pseudocode | All used in refit dispatch pseudocode without definition. Acceptable as "internal helpers" but not contracted. |

### 1.4 Field-level divergences (DIVERGENT entries above, expanded)

#### `AdapterSpec` (MD §5.2 vs DM §6.1)

Same name, completely different shape — and DM acknowledges this by commenting
"not the ML_DATA one" at L574. The two types are unrelated:

| Field | MD §5.2 (data-side) | DM §6.1 (graph-side) |
|---|---|---|
| `id` | str (unique adapter id) | str (e.g. `"sklearn.transformer"`) |
| `version` | str (semver) | str = `"1.0.0"` |
| `input_type` | TypeId | absent |
| `input_representation` | RepresentationId \| None | absent |
| `output_representation` | RepresentationId | absent |
| `output_type` | TypeId | absent |
| `supervised` | bool = False | absent |
| `stateful` | bool = False | absent |
| `lossy` | bool = False | absent |
| `fit_scope` | Literal["none","train_only","fold_train"] | absent |
| `cost_hint` | dict[str, Any] | absent |
| `kind` | absent | NodeKind |
| `priority` | absent | int (lower = higher precedence in `matches` race) |
| `capabilities` | absent | frozenset[str] |

Both names are valid in their own namespace; the convention DM uses
(commenting on the collision) is acceptable for documentation, but at
import time a `from ml_data.contract import AdapterSpec as MlDataAdapterSpec`
will be required somewhere — make this explicit in DM §6 or rename DM's
class to `OperatorAdapterSpec` to remove the collision.

#### `AggregationPolicy` (SRC §2.10.1 vs DM §8.3 — and UC3 §3.3 actual usage)

| Field | SRC (sketch) | DM v1 | UC3 usage | UC6 usage |
|---|---|---|---|---|
| `level` | Literal[...] | absent | `level="sample"` | absent |
| `aggregation_level` | absent | Literal[...] = `"sample"` | absent | absent |
| `method` | Literal[...] = `"none"` | Literal[...] = `"none"` | `method="robust_mean"` | absent |
| `custom_aggregator` | str \| None | absent | absent | absent |
| `keep_observation_predictions` | bool = True | absent | `keep_observation_predictions=True` | absent |
| `exclude_outliers` | absent | absent | `exclude_outliers={enabled, threshold}` | absent |
| `outlier_threshold` | absent | float \| None = None | absent | absent |
| `use_proba` | absent | bool = False | absent | `use_proba=False` |
| `coverage` | absent | Literal[...] = `"drop_incomplete"` | absent | absent |
| `missing_value` | absent | Literal[...] = `"drop"` | absent | absent |
| `duplicate_resolution` | absent | Literal[...] = `"error"` | absent | absent |
| `fold_mismatch` | absent | Literal[...] = `"error"` | absent | absent |
| `validate_oof` | absent | absent | absent | `validate_oof=True` |
| `join_on` | absent | absent | absent | `join_on="sample_id"` |

UC3 and UC6 invoke `AggregationPolicy` with fields that match **neither** SRC
nor DM. They mix `level/method` (SRC), `keep_observation_predictions` (SRC),
`exclude_outliers` (new), `validate_oof` / `join_on` (new). Patch 6 collapses
these into DM's canonical set.

#### `AugmentationPolicy` (MD §9.4 vs DM §9.1)

| Field | MD §9.4 | DM §9.1 |
|---|---|---|
| `apply_to` | Literal[...] = `"train_only"` | Same |
| `inherit_target` | bool = True | Same |
| `inherit_group` | bool = True | Same |
| `forbid_validation_augmentation` | bool = True | Same |
| `store_origin_mapping` | bool = True | Same |
| `seed_scope` | Literal[...] = `"fold"` | Same |
| `multiplier` | absent | int \| None = None |

The only difference is `multiplier`. MD's `AugmentationPlan` (§9.4 / L1392)
already holds `multiplier: int` — so DM's addition is redundant. Patch 5
moves it out.

### 1.5 Defined-but-orphan within the cross-document scope

| Type | Defined in | Notes |
|---|---|---|
| `AlignmentPlan` | MD §6.3 | Used only inside MD §6.4 algorithm; never crosses to DM. DM execution loop (§7.5) treats `align` as opaque. OK in v1 if MD keeps it internal. |
| `CollationPolicy` | MD §8.1 | DM never names it; only mentions "collation" abstractly (DM §18 perf). |
| `ArtifactSerializer` (Protocol) | MD §11.3 | DM owns artifact storage — Protocol is data-side only and only invoked in MD §11.3 narrative. Acceptable. |
| `RefitPlan` | DM §11.3 | Never instantiated; refit strategies bypass it. Either delete or wire into dispatch. |
| `OperatorRegistry.list_for_kind` | DM §6.4 | Defined; used only in resolution algo. OK. |
| `TypeCapability` | MD §4.1 | Used internally by plugin contract; no cross-document refs needed. OK. |

---

## 2. Coherences semantiques

### 2.1 SampleRelation et identifiants (observation / sample / target / group / origin)

- Canonical definition: MD §2.11. Five tuple-parallel arrays: `observation_ids`,
  `sample_ids`, `target_ids` (optional), `group_ids` (optional), `origin_ids`
  (optional, `SampleId | None`).
- **SRC §2.10.1 / L1113** defined a similar dataclass but used `str | int`
  union for `observation_ids`. MD tightened it to `ObservationId = NewType(str)`.
  DM does not redefine but uses raw `str` tuples in `PredictionBlock`
  (DM §8.1 / L779-L780): `observation_ids: tuple[str, ...] | None`,
  `target_ids: tuple[str, ...] | None`. This is **inconsistent with MD's NewTypes**:
  DM should use `ObservationId` and `TargetId` from `ml_data.contract`.
- DM `FoldSet` (§7.1) and `SplitPolicy` (§7.2) reference `split_unit` values
  matching `SampleRelation` fields: `"observation"`, `"sample"`, `"target"`,
  `"group"`. Mapping is **coherent** with MD's nullable rules (DM §7.3 rule:
  if `split_unit="group"` and `group_ids is None`, splitter raises
  `SplitPolicyError` — see UC4 §4.5).
- UC3 §3.3 uses `SplitPolicy(split_unit="target", forbid_origin_cross_fold=True)`
  + `GroupKFold(n_splits=5)` and says "groups = target_ids derives de
  SampleRelation". Mapping `target_id -> group` for the splitter is correct,
  but this **requires DAG-ML to translate `split_unit="target"` into a
  group-key feed for GroupKFold**. The spec does not explicitly describe this
  translation; DM §7.3 only says "the corresponding column is used".
- `origin_id == sample_id` is forbidden by MD §2.11 / L417. DM enforces this
  in §9.1 invariants and §14.7 runtime check. Coherent.
- Verdict: **mostly coherent**. The single fix needed is DM imports of
  `ObservationId` / `TargetId` types instead of raw `str`.

### 2.2 OOF et leakage

- MD exposes the hooks: `SampleRelation.origin_ids`, `SampleRelation.group_ids`,
  `AdapterSpec.fit_scope="fold_train"`. MD strictly delegates OOF/no-leakage
  to DAG-ML (MD §1.2 non-perimeter).
- DM implements OOF via `oof_join` (§8.2), `PredictionJoinNode` (§9.5),
  `AggregationPolicy` (§8.3), runtime check (§14.7) and error taxonomy
  (§14.6: `OOFError` and subclasses).
- **UC3 (repetitions)**: requires `split_unit="target"` and an aggregator node.
  DM §7.3 and UC3 §3.5 both say "obs of one sample go to the same fold". The
  `oof_join` algorithm (DM §8.2) builds the meta table indexed by `sample_id`.
  When predictions are at observation level, DM §9.7 aggregates per
  `aggregation_level` — but the actual aggregation policy in UC3 uses
  `level="sample", method="robust_mean", exclude_outliers={...}`, with field
  names that do not match DM `AggregationPolicy` (`aggregation_level`, `method`,
  `outlier_threshold`). **Field-name drift** — UC3 must be updated to DM names,
  or DM must be extended with `keep_observation_predictions`,
  `exclude_outliers` (boolean + threshold dict).
- **UC5 (augmentation)**: requires augmented copies of a train sample to stay
  on train side. DM §9.1 invariant + DM §14.7 runtime check covers it. UC5 §5.7
  is coherent.
- **UC6 (stacking)**: the `oof_join` algorithm (DM §8.2) handles the typical
  case. UC6's policy field `validate_oof: True` is **not in DM
  `AggregationPolicy`** (only `coverage`, `missing_value`, `duplicate_resolution`,
  `fold_mismatch` exist). Either DM adds it as a no-op (`validate_oof` is
  implicit-by-default) or UC6 should drop the flag.
- **UC11 (`unsafe=True` refused)**: DM §8.5 has `unsafe_use_train=True` and
  records `unsafe_leakage=True` on the lineage. UC11 invents a second flag
  `explicit_confirmed_leakage=True` and adds an `oof_safe: bool` field on
  `PredictionBlock`. **Three divergences**:
  (a) UC11 raises `OOFLeakageError`, DM defines only `OOFUnsafeUsageError`.
  (b) UC11 adds `oof_safe` field; DM `PredictionBlock` (§8.1) does not have it.
  (c) UC11 requires double opt-in; DM §8.5 mentions only a single flag.
- **UC12 (mixed merge)**: covered by `oof_join` algorithmically for the
  prediction part, but the "feature part" of the join is not addressed by
  `oof_join` (which returns a `FeatureTable` from `PredictionBlock`s only).
  The `MixedJoinNode` / `mixed_join` kind to combine `FeatureBlock` +
  `PredictionBlock` is missing from DM. `oof_join` as currently specified
  does **not** suffice for UC12.

### 2.3 Augmentation, OOF et refit

- **Signature**: `AugmentationAdapter` is defined in MD §9.4 (returns
  `tuple[DataBlock, SampleRelation]`). DM §9.1 calls `adapter.transform(...)`
  with the same contract — coherent.
- **AugmentationPolicy is DIVERGENT**: MD §9.4 has 7 fields; DM §9.1 adds
  `multiplier: int | None`. ML_DATA's section explicitly says `apply_to` is
  "passed through" — but `multiplier` is a per-call number, more naturally
  declared per-adapter than as a policy. Decision needed: keep `multiplier`
  in `AugmentationPolicy` (DM choice), or push it into `AugmentationPlan`
  (MD's nested type) which already has `multiplier`.
- **Refit re-application**: UC5 §5.5 + §5.6 (point 1) raises an open question:
  "Refit final reapplique-t-il l'augmentation ?". DM §9.1 says "At REFIT,
  `apply_to in {"train_and_refit", "all_partitions"}` enables augmentation
  on the full train set". UC5 says default = "yes when apply_to='train_only'
  with refit considered as train in absence of val". **The two
  semantics differ**: DM requires explicit `apply_to="train_and_refit"`,
  UC5 wants implicit refit augmentation. Spec must pick one.
- **RefitPlan + augmentation**: DM `RefitPlan` (§11.3) only carries
  `refit_overrides`, no augmentation override. The refit dispatch (DM §11.5)
  re-executes the plan via `execute(...)` which passes `phase=Phase.REFIT` —
  augmentation nodes then re-trigger if their `apply_to` allows. Coherent
  but undocumented as such.

### 2.4 SplitPolicy et split_unit

- `split_unit` values: MD §2.11 implicit (5 id kinds);
  DM `FoldSet.split_unit` (§7.1) = `"observation" | "sample" | "target" | "group"`;
  DM `SplitPolicy.split_unit` (§7.2) = same;
  UC3 uses `"target"`, UC4 uses `"group"`. Coherent set.
- `group_ids` feed to the splitter: MD §9.3 says
  `splitter.split(X, y, groups=relation.group_ids)`. DM §7.3 says
  "DAG-ML synthesises the groups". No conflict.
- Issue: DM §7.3 says **observations of the same sample inherit the fold of
  that sample** when granularity is `per_sample_repeated`. UC3 implements this
  via `split_unit="target"`, not via the observation-broadcast rule (which is
  for `sample`-level splits). The two rules are **complementary** but the
  document tree never spells out when each applies. Recommend a table:

| `split_unit` | Granularity | Mechanism |
|---|---|---|
| `sample` | `per_sample_repeated` | All obs of a sample go to the same fold |
| `target` | `per_sample_repeated` | All samples sharing a `target_id` go to same fold |
| `group` | any | Splitter receives `groups=relation.group_ids` |
| `observation` | any | Splitter sees raw observation ids (rarely safe) |

### 2.5 SeedContext et reproductibilite

- `SeedContext` defined in DM §12.1 / L1369 and **only there**. MD §9.5
  explicitly says it does not implement `SeedContext` and assumes the caller
  passes a derived `random_state: int`. Coherent.
- All UCs use `SeedContext.child(...)`. The hierarchy
  `(root_seed, run_id, variant_id, node_id, fold_id, trial_id, branch_path, aug_index)`
  is in DM §12.1. UC1 §1.5 uses `SeedContext.child(node_id, fold_id)`;
  UC5 §5.5 uses `SeedContext.child(node_id, fold_id)`; UC7 §7.5 uses
  `SeedContext.child(trial_id=k)`; UC10 §10.5 uses `SeedContext.child(node_id="branch:nir_canon")`.
  All consistent with the available `child(**labels)` signature.
- Verdict: OK.

### 2.6 Schema fingerprint et replay

- MD §11.2 defines `schema_fingerprint(schema, fusion, adapter_specs) -> str`.
- DM §5.3 (planning algorithm) calls **`_schema_fingerprint(ctx.dataset_schema,
  ctx.policy, node_plans)`** — different signature (3rd arg is `node_plans`
  vs ML_DATA's `adapter_specs`). The private underscore name suggests an
  internal helper, but the implementation should reuse MD's public function.
  **Fix**: DM should call `ml_data.contract.schema_fingerprint(schema,
  policy.fusion, [np.adapter_id for np in node_plans if np.adapter_id])` —
  passing adapter specs derived from node plans, not the raw plans.
- DM `ExecutionBundle` (§15.1) stores `data_schema: DatasetSchema` but
  the predict replay (§15.2 / L1737) reads `bundle.data_schema_fingerprint`.
  **The field does not exist on the bundle class.** Either rename the access
  or add a precomputed `data_schema_fingerprint: str` to the bundle.
- UC9 §9.4 is consistent with MD's `schema_fingerprint(schema, fusion, adapters)`
  signature.
- Fields entering the fingerprint (MD §11.2):
  - `schema` (sources sorted by id, targets/metadata keys sorted)
  - `fusion: FusionPolicy | None`
  - `adapter_specs` (sorted by id)
- Per UC §D4 open question, this scope is the right level. **What's missing
  in MD**: explicit decision on whether `coordinates` of axes (e.g. exact
  wavelength values) are part of the fingerprint. The current canonical_json
  rule (MD §11.1) serialises every field, so coordinates ARE included by
  default — which is what UC9 §9.4 needs for "weather schema diverge -> reject".
  OK but should be documented explicitly.

### 2.7 DataPlanner.resolve et ModelInputSpec

- MD §7.3 signature:
  ```python
  resolve(self, dataset: MLDataset, sources: Sequence[SourceId],
          model_input: ModelInputSpec, policy: FusionPolicy) -> DataPlan
  ```
- DM §5.3 call site:
  ```python
  ctx.data_planner.resolve(dataset=None, sources=..., model_input=...,
                            policy=ctx.policy.fusion)
  ```
- **CONFLICT**: DM passes `dataset=None` ("schema-level resolution"). MD says
  `dataset: MLDataset` (non-optional). Two options:
  1. Make MD's `dataset` parameter `MLDataset | DatasetSchema | None`, with
     schema-level resolution explicit. Document the modes.
  2. Add a sibling `resolve_from_schema(schema, ...)` in MD §7.3.
  The first is simpler; the second is cleaner.
- DM never owns a `DataPlanner` instance. `PlanningContext.data_planner`
  (DM §5.1) carries it. The construction site is not documented — likely
  injected by the application layer at `run_context.data_planner`. UC1 §1.7
  implies nirs4all owns it. **Fix**: DM should add a one-paragraph note on
  how a `DataPlanner` is constructed and injected (factory? default impl?).
  The current spec leaves this as an implicit dependency.

### 2.8 FeatureJoiner comme adapter standard

- MD §6.5 explicitly says: "The `FeatureJoiner` is a standard adapter that
  produces a stable schema at fit time".
- DM §9.4 defines `FeatureJoinNode` as a first-class DAG node (NodeKind
  `FEATURE_JOIN`). The spec says "horizontal concat of `FeatureTable`s"
  and that the node "wraps an ML_DATA `FeatureJoiner` with a fixed schema
  captured at FIT_CV".
- The DataPlan `join` step (MD §7.2 + §7.4 phase 3) calls
  `adapter_id="fusion.feature_joiner"` — a DataPlan step internal to the
  DataPlanner. So there are TWO places where joining happens:
  1. Inside a DataPlan (for multi-source fusion at the boundary of a single
     model node);
  2. As a parent-level `FeatureJoinNode` (for joining branch outputs in
     stacking-style pipelines).
- This is **coherent but subtle**: the DataPlan's join handles "early fusion
  to feed a single model"; the FeatureJoinNode handles "branch reassembly".
  Implementation should reuse the same `FeatureJoiner` adapter underneath.
  Recommended: cross-reference in MD §6.5 ("for the DAG-ML-level use see
  DM §9.4 `FeatureJoinNode`").

### 2.9 PredictionBlock et stockage

- DM owns `PredictionBlock` (§8.1) and `PredictionStore` (§8.1). MD does not
  re-define them — coherent with the frontier (DM §1.3 says "DAG-ML never
  passes a `PredictionBlock` to ML_DATA").
- The MD frontier diagram and §13 "Non-buts explicites" both confirm
  PredictionStore is DAG-ML's. **No contradiction**.

### 2.10 NodeKind enum coverage vs DSL keywords

DM `NodeKind` enum (DM §2.1):
```
TRANSFORM, Y_TRANSFORM, SPLIT, MODEL, FORK, MAP, FEATURE_JOIN,
PREDICTION_JOIN, SOURCE_JOIN, TAG, EXCLUDE, AUGMENTATION, ADAPTER,
TUNER, SUBGRAPH, CHART
```

UCs and Annexe A reference these additional pseudo-kinds:
`materialize`, `adapt`, `align`, `join`, `prediction`, `sample_rel`,
`aggregator`, `mixed_join`, `invariant_check`, `restructure`, `refit`,
`search_space`, `variant`. Some of these are sub-step labels inside a
`DataPlan` (MD §7.2: `materialize`, `adapt`, `align`, `join`, `collate`)
and should not appear as node kinds at the DAG-ML level. Others are
genuine missing kinds:

| Pseudo-kind | UC site | Should it be a NodeKind? | Recommendation |
|---|---|---|---|
| `materialize` | UC1 §1.4 node table | No, DataPlanStep kind | Document this in UC1 |
| `adapt` | UC1, UC2, UC4, UC5 | No, DataPlanStep kind OR wraps ADAPTER node | Use `ADAPTER` NodeKind in UC node tables |
| `align`, `join` | UC1 §1.4 | No, DataPlanStep kind | Document |
| `sample_rel` | UC4 §4.4 | No, fact-of-life from MD | Drop from node table |
| `aggregator` | UC3 §3.4 | YES — distinct from MODEL | Add `AGGREGATOR` |
| `mixed_join` | UC12 §12.4 | YES — extends PREDICTION_JOIN | Add `MIXED_JOIN` |
| `invariant_check` | UC12 §12.4 | NO — runtime check, not a node | Drop from node table |
| `restructure` | UC Annexe A | YES if Q17 picks NodeKind | Add `RESTRUCTURE` |
| `prediction` | UC1 §1.4 | No, this is a port kind, not a node | Drop |
| `search_space`, `variant`, `refit` | UC7 §7.4, UC8 §8.4 | No, these are phases/structures | Drop |

### 2.11 SubgraphNodeSpec et reification

- DM §2.3 / L185 defines `SubgraphNodeSpec` with `inline_policy: Literal["inline",
  "opaque", "auto"]`. SRC §2.10.7 / L1368 has the same field. UC10 uses both
  `"opaque"` and `"inline"` explicitly. Coherent.
- DM `GraphInterface` (§2.3) and SRC `GraphInterface` (§2.10.7) have the same
  signature (`inputs`, `outputs` tuples of `PortSpec`).
- UC10 §10.5 invariant "GraphInterface doit etre declaree" is correctly
  reflected in DM §2.3 `GraphSpec.interface: GraphInterface` (required field,
  not optional).
- Verdict: OK. Note that UC10 §10.6 (point 1) raises an open question on
  whether opaque sub-DAGs propagate their `SearchSpace` to the parent tuner.
  DM has no answer; the spec says "scope ferme" in UC10 §10.6 but the type
  system does not enforce this.

---

## 3. Use cases vs specifications

| UC | Couverture | Manques | Severite |
|---|---|---|---|
| UC1 (multi-source RF) | Algorithms exist (DataPlan resolve, FeatureJoiner). | DSL keywords `{"sources":...}`, `{"by_source":...}`, `{"fusion":...}`, `{"split_policy":...}` not in DM §4.3 lowering table. `requires_user_choice` escalation policy under-specified for the UC1 friction (image_embedding lossy default). | YELLOW |
| UC2 (multi-instrument) | Variant A (stack_channels) is supported by `FusionPolicy.mode="stack_channels"`. Variant B (by_source branches) is supported by ForkNode separation. | The PLS-3D adapter (UC2 §2.4 ASCII art) requires a `pytorch.module` rank=3 input or a custom POP-PLS model adapter — neither is shipped by DM §6.5. UC2 §2.6 (point 3) explicitly flags this. `{"by_source":...}` top-level DSL keyword absent from DM §4.3. | YELLOW |
| UC3 (repetitions + agg) | Folds with `split_unit="target"` covered; observation-level fit + sample-level aggregation pattern documented in DM §17 UC4 of the spec. | `AggregationPolicy` field names diverge (UC3 uses `level`, `method`, `keep_observation_predictions`, `exclude_outliers`; DM has `aggregation_level`, `method`, ...). `AggregatorNode` is not a formal NodeKind. `{"aggregate":...}` not in lowering table. | RED |
| UC4 (group split) | `SampleRelation.group_ids` + `SplitPolicy(split_unit="group")` covered. | `{"split_policy":...}` DSL keyword not in §4.3. Edge case (UC4 §4.6 point 1) "patient with 2 mesures" not addressed by the spec. | YELLOW |
| UC5 (augmentation train-only) | DM §9.1 covers the invariants. | `RefitPlan` does not carry augmentation override. UC5 §5.5 "refit reapplique-t-il l'augmentation" ambiguous (DM says "only if `apply_to='train_and_refit'`"; UC5 implies default behaviour). `Mixup`-style augmentation (UC5 §5.6 point 2) needs `origin_ids` list, not scalar — current schema is scalar `tuple[SampleId | None, ...]`. | YELLOW |
| UC6 (stacking) | Fully covered by `oof_join`, `PredictionJoinNode`, refit_stacking. | UC6 §6.3 uses `policy.unsafe=False, validate_oof=True, join_on="sample_id"` — `validate_oof` and `join_on` not in DM `AggregationPolicy`. The refit_stacking pseudocode (DM §11.5) is correct conceptually but uses helpers (`leaf_models_of`, `meta_stacker_of`) that are not defined. | YELLOW |
| UC7 (search + tuner) | `SearchSpace`, `Variant`, `TunerAdapter`, `TuningNodeSpec` all present. | `{"tuner":...}` top-level DSL keyword not in §4.3 (only `_or_`, `_grid_`, ... are listed). `_chain_` semantics interaction with `_sample_` (UC7 §7.3) under-specified. `early_stopping` extras in `TuningNodeSpec.early_stopping` (DM §10.5) unspecified — UC7 §7.6 point 2 explicitly raises this. | YELLOW |
| UC8 (branches by metadata) | `ForkPolicy(mode="separation", by="metadata")` covered. | `{"by_branch":...}` DSL keyword absent from §4.3. UC8 §8.6 point 2 ("new site D at predict") not addressed: no fallback mechanism. `{"merge": "concat"}` is mentioned in §4.3 as "FeatureJoinNode in separation mode (reassembly)" but the actual code-path is unclear: concat is a prediction-level reassembly, FeatureJoin is feature-level. Conflict between §4.3 row and UC Annexe A row for `"concat"`. | YELLOW |
| UC9 (predict from bundle) | DM §15.2 covers replay; MD §11.2 covers fingerprint. | `bundle.data_schema_fingerprint` accessed but not stored on the bundle (see §2.6). `schema_check="strict"/"compatible"/"loose"` (UC9 §9.3) is not in DM's public `predict()` signature (DM §16). | RED |
| UC10 (subgraph reification) | `SubgraphNodeSpec`, `GraphInterface` present. | UC10 §10.6 point 3 ("auto inline criterion") undocumented in DM §2.3. UC10 §10.6 point 4 ("cross-version subgraphs") not addressed in `plugin_versions` / `requires_plugin_versions`. | YELLOW |
| UC11 (unsafe refused) | DM §8.5 covers the basic refusal. | `OOFLeakageError` name (used in UC) vs `OOFUnsafeUsageError` (in DM taxonomy). `oof_safe` field on `PredictionBlock` not declared. `explicit_confirmed_leakage` second-flag mechanism not in DM. | RED |
| UC12 (mixed merge) | `oof_join` handles prediction part. | `MixedJoinNode` not defined. `FeatureBlock` type referenced but not declared in MD or DM v1. `block_kind` discriminator not in type system. `{"merge": "all"}` policy fields `branches_as_features` / `branches_as_predictions` not specified. | RED |

### 3.1 DSL keyword coverage matrix

DM §4.3 lists 21 lowering rules. UCs introduce additional top-level dict
keywords not in that table:

| Keyword (as used in UCs) | DM §4.3 row? | UC sites | Action |
|---|---|---|---|
| `{"sources": [...]}` | NO (only as `SourceJoin` via `{"merge":{"sources":...}}`) | UC1, UC2, UC3, UC4, UC5, UC6, UC7, UC8, UC10, UC11, UC12 | Add row: declares input sources for the graph; resolved into `GraphSpec.inputs` ports |
| `{"by_source": {...}}` | NO | UC1, UC2, UC4 | Add row: per-source preprocessing dict; lowers into per-source ADAPTER subgraphs |
| `{"by_branch": {...}}` | NO | UC8 | Add row: per-branch model dict; lowers into separation MapNode subgraphs |
| `{"fusion": FusionPolicy(...)}` | NO (only as part of `source_join` inline) | UC1, UC2, UC4 | Add row: top-level fusion declaration; sets `PlanningPolicy.fusion` for the downstream model node |
| `{"split_policy": SplitPolicy(...)}` | NO | UC1, UC3, UC4 | Add row: configures the next `SPLIT` node |
| `{"sample_relation": {...}}` | NO (only used in DM §17 UC4 example) | DM §17, none in UC doc | Same as `split_policy`; choose one and drop the other |
| `{"aggregate": AggregationPolicy(...)}` | NO | UC3 | Add row: appends an `AGGREGATOR` NodeKind (see Q10 / Patch 9) |
| `{"tuner": {...}}` | NO | UC7 | Add row: appends a `TUNER` node controlling upstream `SearchSpace` |
| `{"adapter": "<id>", "params": {...}}` | NO (no explicit row; the bare-dict adapter form is implicit) | UC1, UC2, UC3, UC4, UC5, UC10, UC12 | Add row: lowers into `ADAPTER` NodeKind, resolved via ML_DATA `AdapterRegistry` |
| `{"source_join": {...}}` (top-level) | NO (only as a `merge` sub-mode) | DM §17 UC2 / L1976 | Confirm canonical form: top-level dict OR `{"merge": {"sources": ...}}` (DM §4.3); pick one |
| `{"branch": {"by_source": True, "steps": {...}}}` | YES (row "by_X") | UC2 §2.3 | Coherent with `ForkPolicy(mode="separation", by="source")` |

Without these rows in DM §4.3 the Compiler does not know how to lower the
UCs. Patch 11 (new): add the missing rows in a single block.

---

## 4. Gaps importants non couverts

The following gaps would block a v1 implementation:

- **Multi-source models without concat (UC2 stack_channels, UC1 dict_input).**
  DM §1 mentions multi-source compatibility; MD §6.2 `FusionPolicy.mode` has
  `dict_input` / `list_input`. But: a model adapter declaring
  `multi_source=True` on its port (MD §7.1 `InputPortSpec.multi_source`) is
  routed through MD §7.4 phase 3, which builds an align+join step OR passes
  per-source outputs through unchanged. **No example or test in MD/DM clearly
  spells out the dict_input handoff** — UC1 mentions `RandomForest` accepts
  multi_source via concat, but a true multi-input PyTorch network is never
  exercised. For v1 this can be deferred IF an explicit clarification is
  added that DM `ModelAdapter.fit(inputs: dict[str, DataBlock], ...)` accepts
  one entry per source-port for the multi-input case.

- **Chart / visualization nodes at PREDICT.** DM §2.1 `NodeKind.CHART`
  is mentioned in §4.3 as "side-effect node, no outgoing edges". `supported_phases`
  for `dagml.chart` adapter (DM §6.5) says "any" — but at PREDICT a chart
  has no obvious purpose. UC tables (Annexe F) do not exercise EXPLAIN for
  any UC (all `-`), and CHART has zero UC coverage. **For v1, define: CHART
  is supported only in FIT_CV/REFIT/SELECT (skipped at PREDICT/EXPLAIN)**, or
  declare it as an entirely separate concern.

- **`{"restructure"}` (rep_to_sources, rep_to_pp).** DM §4.3 lists them as
  "data layout directive; not a DAG node, resolved at PLAN via DataView.extra".
  UC Annexe A contradicts this by mapping them to `restructure` NodeKind.
  Today nirs4all has dedicated `RepetitionController` + flow controllers in
  `controllers/flow/`. **For v1 either reify them as a `RESTRUCTURE`
  NodeKind in DM or commit to the "DataView.extra" approach** and detail
  how layout is changed without a node. Current ambiguity blocks the
  Compiler implementation.

- **Mixed merge type discriminator (UC12).** Without `FeatureBlock` declared
  in MD and a `MixedJoinNode` declared in DM, UC12 cannot be implemented.
  Smallest fix: extend `oof_join` to accept a mix of `FeatureTable`
  (for feature branches) and `PredictionBlock` (for prediction branches),
  controlled by per-branch flags. Larger fix: introduce `FeatureBlock` and
  `MixedJoinNode`.

- **nirs4all-specific controllers reification.**
  - `sample_augmentation`: covered by `AUGMENTATION` NodeKind.
  - `tag`: covered by `TAG` NodeKind.
  - `exclude`: covered by `EXCLUDE` NodeKind.
  - `concat_transform`: DM §4.3 says "FeatureJoinNode of transformed branches".
    Implementation needs an internal sub-fork pattern.
  - `feature_augmentation`: DM §4.3 says "lowered into AdapterNode + FeatureJoinNode".
    Currently sketchy — not exercised in any UC.
  - `rep_to_sources` / `rep_to_pp`: see gap above.
  - `AutoTransferPreprocController`: nowhere in DM/MD — needs a decision
    on whether this becomes a transform adapter or a `RESTRUCTURE` kind.

- **Dry-run / plan-only mode.** Neither MD nor DM ship a "plan inspector"
  that returns the DataPlan + ExecutionPlan without execution. MD §7.6
  explicitly raises this as an open question (`DataPlanner.dry_run`).
  For interactive UI development this is required. **For v1 add a
  `dagml.plan_only(pipeline, dataset, policy) -> ExecutionPlan` function**.

- **`CacheConfig` reification.** nirs4all's existing `CacheConfig` (with
  `step_cache_enabled`, `step_cache_max_mb`, `use_cow_snapshots`,
  `log_cache_stats`, `memory_warning_threshold_mb`) is referenced once in
  DM §13.4 without a class definition. **Implementation will need it**;
  either define in DM §13 or drop the reference.

- **Logging / verbose levels.** SRC §1 and UC8 mention `verbose=1`/`verbose=2`.
  DM §14.8 defines `MetricsLogger` (scalar/histogram) but no structured
  console logger. Acceptable for v1 if the application layer (nirs4all)
  owns it.

- **PredictionStore concurrency.** DM §14.4 says "each worker gets its own
  PredictionStore (store=None for in-memory)" — same pattern as nirs4all's
  orchestrator. But the PredictionStore Protocol (§8.1) does not document
  thread-safety or merge semantics. **Add an explicit clause**: implementations
  must be thread-safe OR the executor must serialise writes when a single
  store is shared.

- **Multi-output / multi-target regression.** MD §16 question 2 explicitly
  defers this. UC table makes no multi-target case. Acceptable defer.

- **Streaming / lazy datasets.** MD §16 question 1 + DM §21 question 1 both
  defer. Acceptable.

- **EXPLAIN coverage.** UC Annexe C shows EXPLAIN as `-` for every UC. The
  `ExplainerAdapter` (DM §15.4) is defined but no UC validates the contract.
  For v1, at minimum a "UC1 + SHAP" walk-through would lock the design.

---

## 5. Decisions design transverses (cross-document)

Consolidation of D1-D5 (UC), DM §21, MD §16 + MD §7.6.

| ID | Question | Documents | Type | Bloquant v1? |
|---|---|---|---|---|
| Q1 | Auto-resolve lossy adapter chains: `allow_lossy_adapters=True` defaults to auto, or always escalate via `requires_user_choice`? | UC §D1, MD §7.6 | api | NO (default to "escalate when multiple lossy paths") |
| Q2 | Ranking level (obs / sample / group / branch) when predictions exist at multiple levels: pick most aggregated by default, override via `RankingPolicy.ranking_level`. | UC §D2 | api | NO (defer override) |
| Q3 | Refit hyperparams: reuse best fold's, aggregate across folds, or re-tune on full train? | UC §D3 | exec | NO (reuse best fold by default) |
| Q4 | Schema fingerprint scope: include axis coordinates, feature names, plugin versions; exclude seeds. | UC §D4, MD §11.2, DM §15.2 | data | YES (coordinates inclusion needs explicit statement in MD §11.2; fingerprint field on bundle must be added) |
| Q5 | Parallelism granularity (variant / fold / branch / inline subgraph) and nested-parallelism control: single budget at RunContext level. | UC §D5, DM §14.4 | perf | NO (DM §14.4 already states this; nail down precedence) |
| Q6 | Streaming / lazy `DataBlock` (dask / zarr backend): defer to v1.1. | MD §16 #1, DM §21 #1 | data | NO |
| Q7 | Joint adapters (cross-source single adapter, e.g. CCA between NIRS and image embedding): defer to v1.1. | MD §16 #4, DM §21 #2 | data | NO |
| Q8 | Multi-target sources (rank-2 `y` natively or via multiple `TargetBlock`s): defer. | MD §16 #2, DM §21 #3 | data | NO |
| Q9 | Partial-fit / incremental learning: out of scope v1. | MD §16 #3, DM §21 #4 | exec | NO |
| Q10 | Aggregation adapter for observation→sample reduction: kept in DAG-ML's prediction layer (not in MD). | MD §16 #5 | api | YES (UC3 §3.3 uses `AggregationPolicy` fields not defined in DM; must align names) |
| Q11 | `DataPlanner.dry_run(plan, dataset, view)`: defer to v1.1. | MD §7.6 | api | YES (interactive UI needs at least a plan-only entry; see §4 gap) |
| Q12 | `requires_user_choice` structured alternatives: defer to v1.1. | MD §7.6 | api | NO |
| Q13 | Soft / hard constraints in tuning (max param count, max time): kept as `TunerAdapter` extras. | DM §21 #5 | exec | NO |
| Q14 | `unsafe=True` opt-in: single flag (DM §8.5) or double confirmation (UC11 `explicit_confirmed_leakage`)? | DM §8.5, UC11 | api | YES (UI / programmatic semantics differ; pick one and update both docs) |
| Q15 | `OOFLeakageError` vs `OOFUnsafeUsageError`: pick one error name. | DM §14.6, UC11+UC5 | api | YES (trivial doc fix but currently inconsistent across files) |
| Q16 | `oof_safe: bool` field on `PredictionBlock`: add or derive at runtime from `partition`+fingerprint? | DM §8.1, UC11 | data | YES (UC11 invariant needs the field or an equivalent runtime check) |
| Q17 | `rep_to_sources` / `rep_to_pp`: reified as `RESTRUCTURE` NodeKind (UC Annexe A) or as `DataView.extra` directives (DM §4.3)? | DM §4.3, UC Annexe A | exec | YES (contradiction blocks Compiler design) |
| Q18 | Augmentation re-application at REFIT: implicit (default) or explicit (`apply_to="train_and_refit"`)? | DM §9.1, UC5 §5.5 | exec | YES (different defaults in different documents) |

Duplicates / collapsed:
- UC §D1 ≈ MD §7.6 (lossy escalation) -> Q1.
- DM §21 #1 + MD §16 #1 (streaming) -> Q6.
- DM §21 #2 + MD §16 #4 (joint adapters) -> Q7.
- DM §21 #3 + MD §16 #2 (multi-target) -> Q8.
- DM §21 #4 + MD §16 #3 (partial fit) -> Q9.

Direct contradictions (the documents pencher en directions opposees):
- **Q14**: DM §8.5 single flag; UC11 double flag. Pick.
- **Q15**: DM `OOFUnsafeUsageError`; UC `OOFLeakageError`. Pick.
- **Q17**: DM §4.3 "not a DAG node"; UC Annexe A says `restructure` NodeKind. Pick.
- **Q18**: DM §9.1 explicit `apply_to="train_and_refit"`; UC5 §5.5 implicit. Pick.

---

## 6. Recommandations de patch

Concrete, minimal patches (priority order). Each is a section-level edit, not
a rewrite.

1. **MD §7.3 — Adjust `DataPlanner.resolve` signature.** Change `dataset: MLDataset`
   to `dataset: MLDataset | None`, OR add a sibling method
   `resolve_from_schema(schema, sources, model_input, policy) -> DataPlan`.
   Today DM §5.3 plans against a schema (no materialised dataset) and the
   signature mismatch makes the spec un-implementable as written. Same patch
   adds a sentence: "How is a `DataPlanner` instance obtained? It is injected
   via `RunContext.data_planner`. The application layer (e.g. nirs4all)
   constructs a default impl."

2. **DM §15 — Fix `ExecutionBundle.data_schema_fingerprint` access.** Either
   add a field `data_schema_fingerprint: str` to the `ExecutionBundle`
   dataclass (DM §15.1), OR change the predict pseudocode (DM §15.2 / L1737)
   to call `schema_fingerprint(bundle.data_schema, bundle.policy.fusion,
   adapter_specs_from(bundle))`. Add a `policy: PlanningPolicy` field on
   the bundle so the predict path has all inputs.

3. **DM §8 + §14.6 — Reconcile error names.** Either rename
   `OOFUnsafeUsageError` to `OOFLeakageError` in DM §14.6, OR update UC5 §5.5,
   UC11 (all occurrences), UC Annexe E to use `OOFUnsafeUsageError`. Also
   add `LeakageError` (augmentation case) as a new subclass of `OOFError` in
   DM §14.6 to legitimise UC5's reference.

4. **DM §8.1 + §8.5 — Add `oof_safe: bool` field and clarify unsafe flag.**
   Either:
   (a) Add `oof_safe: bool = True` to `PredictionBlock` (default True for
       `partition="val"`; False for `partition="train"` or when fold leak),
       and clarify DM §8.5 to enforce that `PredictionJoin` requires
       `all(b.oof_safe for b in inputs)`; OR
   (b) Derive `oof_safe` at runtime from `partition` (no schema change)
       and explicitly close UC11's expectation of a stored field. Pick (a)
       for explicit auditability. Resolve Q14 (single vs double flag) at
       the same time and remove the contradiction.

5. **MD §9.4 + DM §9.1 — Unify `AugmentationPolicy`.** Move `multiplier` out
   of `AugmentationPolicy` into `AugmentationPlan` (MD §9.4 already has
   `multiplier: int` in `AugmentationPlan`), then drop `multiplier` from
   DM's `AugmentationPolicy`. Add an explicit table for REFIT behaviour
   (Q18): default for `apply_to="train_only"` is "do not re-augment at
   refit"; `"train_and_refit"` re-augments.

6. **DM §8.3 — Extend `AggregationPolicy`.** Either rename / add the
   following fields used by UC3 and UC6:
   - `validate_oof: bool = True` (no-op default; explicit refusal of train preds is handled in §8.5).
   - `keep_observation_predictions: bool = False`.
   - `exclude_outliers: dict[str, Any] | None = None` (`{enabled, threshold}`) — or split into two fields `exclude_outliers: bool = False`, `outlier_threshold: float | None = None`.
   - `join_on: Literal["sample_id", ...] = "sample_id"` (UC6).
   Then update UC3 §3.3 / UC6 §6.3 to use the canonical field names.

7. **DM §2.1 + §4.3 — Resolve `restructure` NodeKind contradiction (Q17).**
   Add `RESTRUCTURE = "restructure"` to the `NodeKind` enum; update DM §4.3
   row for `rep_to_sources` / `rep_to_pp` to say "lowered into `RESTRUCTURE`
   node" (not "data layout directive resolved at PLAN"). This matches
   UC Annexe A and unblocks the Compiler.

8. **MD §2.1 + DM §7.1 + §8.1 — Type alias coherence.** Remove DM's
   `SampleIdT = str` (DM §7.1 / L715). Replace all DM usages with
   `from ml_data.contract import SampleId, ObservationId, TargetId`.
   Update `PredictionBlock` to use `SampleId`, `ObservationId`, `TargetId`
   instead of raw `str`.

9. **DM §2.1 — Extend `NodeKind` for UC12 + UC3.** Add at minimum:
   `MIXED_JOIN = "mixed_join"` (UC12), `AGGREGATOR = "aggregator"` (UC3).
   For UC12 also declare `FeatureBlock` (or alias to `DataBlock(repr=
   tabular_numeric)`) in MD §2 and update the `MixedJoinNode` execution
   contract to accept the union with a `block_kind` discriminator.

10. **DM §16 — Define `CVResult`, `RunResult`, `RefitResult` dataclasses.**
    Three return types are used in the public API (DM §16) but never declared.
    Add a single block at the start of §16 with their fields:
    ```python
    @dataclass(frozen=True)
    class CVResult:
        variant_id: str
        fold_metrics: dict[str, dict[str, float]]   # fold_id -> metric -> value
        oof_predictions: tuple[PredictionBlock, ...]
        warnings: tuple[str, ...] = ()

    @dataclass(frozen=True)
    class RefitResult:
        selected: SelectedGraph
        refit_artifacts: dict[str, tuple[ArtifactRef, ...]]
        refit_predictions: tuple[PredictionBlock, ...]

    @dataclass(frozen=True)
    class RunResult:
        cv_results: tuple[CVResult, ...]
        selected: SelectedGraph
        refit: RefitResult
        bundle: ExecutionBundle
    ```
    Without this the public API is un-typeable and `dagml.run()` cannot be
    consumed by static type checkers.

11. **DM §4.3 — Add the 9 missing DSL keyword rows.** Lowering rules for
    `{"sources":...}`, `{"by_source":...}`, `{"by_branch":...}`,
    `{"fusion":...}`, `{"split_policy":...}`, `{"aggregate":...}`,
    `{"tuner":...}`, `{"adapter":...}`, and a definitive
    `{"source_join":...}` row (see §3.1 above). Without these the Compiler
    has no formal mapping for ~half of the UC DSLs.

Stop. Patches 1-11 are sufficient to unblock implementation of all 12 UCs
modulo the Q6-Q9 deferred items.

---

## 7. Verdict global

The three v1 specs cover the structural problem space well: types, phases,
folds, OOF, refit, bundles, replay, and the ML_DATA / DAG-ML frontier are all
articulated. The four documents are mutually consistent on **most** type
definitions and the high-level frontier is clean (ML_DATA = data, DAG-ML =
graph + invariants + execution).

Remaining issues are concrete and surgical: a handful of type duplications
(`AugmentationPolicy`, `AggregationPolicy`, `AdapterSpec`), one signature
mismatch (`DataPlanner.resolve`), one missing bundle field
(`data_schema_fingerprint`), one error-name divergence (`OOFLeakageError`),
and three formally-undeclared concepts that the UCs need
(`FeatureBlock`, `MixedJoinNode`, `AggregatorNode`). Most can be fixed in
under 50 lines of edits per document.

Two decisions remain genuinely blocking and should be made before code:
Q14 (single vs double unsafe flag), Q17 (`restructure` as NodeKind or
DataView.extra). Both are policy choices, not architectural.

Score: **7.5 / 10**. Specs are ready for a skeleton implementation once
patches 1-11 land; the deferred questions Q6-Q9 will not block v1 work as
long as the public types remain forward-compatible (frozen dataclasses with
default fields, per MD §12.3).

---

End of cross-review v1.
