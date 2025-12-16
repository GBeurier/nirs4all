# ⚠️ INCORPORATED INTO V3 DESIGN

> **This analysis has been incorporated into the V3 artifact system design.**
>
> The root cause analysis in this document directly informed the V3 design:
> - The "Operator Chain" concept in V3 addresses the "branch substeps not recorded" issue
> - V3's `TraceRecorderV3` with chain/branch stacks implements the recommended fix architecture
> - All proposed fixes are subsumed by the V3 redesign
>
> See:
> - [ARTIFACT_SYSTEM_CURRENT_STATE.md](./ARTIFACT_SYSTEM_CURRENT_STATE.md) - Section 4: Execution Trace System
> - [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) - Section 5: Revised Trace Recording
>
> This document is preserved as a historical reference for the analysis that motivated V3.

---

# Multisource + Branching Reload: Deep Analysis (ARCHIVED)

**Date**: December 14, 2025
**Status**: Archived - Superseded by V3 Design (December 15, 2025)

---

## Root Cause Summary (Confirmed in V3 Analysis)

**Primary Root Cause:**
The `BranchController.execute()` method does not record individual substeps in the execution trace. All artifacts created inside branches are lumped under the parent "branch" step without branch_path context.

**V3 Solution:**
The V3 design introduces:
1. **OperatorChain** - Full execution path for every artifact
2. **TraceRecorderV3** with chain/branch stacks - Proper substep recording
3. **Chain-based artifact identification** - No more artifact ID parsing hacks

---

## How V3 Addresses Each Issue

| Issue from Analysis | V3 Solution |
|---------------------|-------------|
| Branch substeps not recorded | `TraceRecorderV3.enter_branch()` / `exit_branch()` with substep recording |
| Artifact ID parsing workaround | `OperatorChain` includes branch_path natively |
| Operation counter divergence | Chain-based identification replaces counter-based naming |
| MetaModel signature mismatch | `MetaModelControllerV3` uses chain-based dependencies |

---

## Original Analysis

*The following is the original analysis document, preserved for historical reference. The proposed "Option A" (Record Branch Substeps) is what V3 implements, with the additional enhancement of the Operator Chain concept.*

[Original analysis content preserved below for reference]

---

### 1. The Problem: Execution Trace Does Not Track Branch Substeps

During training, the pipeline execution flow for branching is:

```
Step 1: MinMaxScaler (X transformer)  → recorded with branch_path=[]
Step 2: MinMaxScaler (Y transformer)  → recorded with branch_path=[]
Step 3: ShuffleSplit                  → recorded with branch_path=[]
Step 4: Branch Controller             → recorded with branch_path=[]
  ├─ Substep: SNV (branch_0)         → NOT recorded (artifacts lumped into Step 4)
  └─ Substep: SavGol (branch_1)      → NOT recorded (artifacts lumped into Step 4)
Step 5: PLSRegression (branch_0)      → recorded with branch_path=[0]
Step 5: PLSRegression (branch_1)      → recorded with branch_path=[1]
```

**The issue:** Step 4 (branch) is recorded as a single step with **ALL branch artifacts** listed together. The execution trace has NO way to know which artifacts belong to which branch.

**V3 Fix:** Branch substeps are now recorded individually with their `branch_path`, and the `OperatorChain` captures the full path through the pipeline.

---

## V3 Implementation Status

See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) Section 13 for implementation phases.
