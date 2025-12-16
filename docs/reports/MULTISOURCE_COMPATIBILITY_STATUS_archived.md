# ⚠️ INCORPORATED INTO V3 DESIGN

> **This document has been incorporated into the V3 artifact system design.**
>
> The issues documented here informed the V3 design decisions. See:
> - [ARTIFACT_SYSTEM_CURRENT_STATE.md](./ARTIFACT_SYSTEM_CURRENT_STATE.md) - Section 7: Known Edge Cases and Failures
> - [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) - Section 1.2: Edge Cases That Must Work
>
> This document is preserved as a historical reference showing the test results that motivated V3.

---

# Multisource + Complex Features Compatibility Report (ARCHIVED)

**Date**: December 14, 2025
**Status**: Archived - Incorporated into V3 Design (December 15, 2025)

---

## Summary of Issues (All Addressed in V3)

| Issue | Root Cause | V3 Solution |
|-------|------------|-------------|
| Branching + Multisource reload fails | Artifact ID doesn't include source_index | V3 OperatorNode includes `source_index` |
| Sklearn Stacking reload fails | Same as above | V3 chain tracking |
| MetaModel signature mismatch | Missing `custom_name` parameter | Fixed in V3 MetaModelController |
| Branch metrics NaN | Y processing in branch context | V3 proper branch substep recording |

---

## Original Test Results

*These tests should all pass after V3 implementation.*

| Test | V2 Status | V3 Target |
|------|-----------|-----------|
| test_basic_branching_multisource | ⚠️ Partial (NaN metrics) | ✅ |
| test_branching_multisource_reload | ❌ Broken | ✅ |
| test_sklearn_stacking_multisource_reload | ❌ Broken | ✅ |
| test_metamodel_multisource | ❌ Broken | ✅ |
| test_metamodel_with_branches_multisource | ❌ Broken | ✅ |

---

## V3 Design Response

The V3 design addresses each issue:

1. **OperatorNode.source_index**: Multi-source transformers are now uniquely identified
2. **Branch substep recording**: Each branch substep is recorded individually in the trace
3. **Chain-based identification**: Artifact lookup uses full operator chain, not name matching
4. **MetaModelController updates**: Signature matches base class, uses chain for dependencies

See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) for implementation details.
