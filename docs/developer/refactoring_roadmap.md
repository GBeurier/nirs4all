# nirs4all Pipeline Refactoring Roadmap

**Status**: Proposed
**Last Updated**: 2024
**Owner**: Development Team

---

## Overview

This document provides a **comprehensive roadmap** for refactoring the nirs4all pipeline infrastructure. The refactoring addresses two critical anti-patterns:

1. **PipelineRunner God Class** (1050 lines, 20+ responsibilities)
2. **Context Dict Chaos** (176+ usage locations, mixed concerns)

These refactorings are **interdependent** and must be executed in the correct sequence.

---

## Related Documents

| Document | Purpose | Status |
|----------|---------|--------|
| [CONTEXT_REFACTORING_PROPOSAL.md](CONTEXT_REFACTORING_PROPOSAL.md) | **CRITICAL BLOCKER** - Context architecture | ⚠️ Must complete first |
| [RUNNER_REFACTORING_PROPOSAL.md](RUNNER_REFACTORING_PROPOSAL.md) | Runner decomposition | 🔒 Blocked by context |
| [RUNNER_ARCHITECTURE_DIAGRAM.md](RUNNER_ARCHITECTURE_DIAGRAM.md) | Visual architecture | Reference |
| [RUNNER_CODE_EXAMPLES.md](RUNNER_CODE_EXAMPLES.md) | Implementation examples | Reference |

---

## Critical Path

```
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: Context Refactoring (3 weeks)                           │
│ ✅ Prerequisite for all other work - CLEAN BREAKING CHANGE       │
├──────────────────────────────────────────────────────────────────┤
│ Week 1: Implement new context module with full type safety       │
│ Week 2: Atomic migration of all 176+ usage locations             │
│ Week 3: Validation, testing, performance benchmarks              │
└──────────────────┬───────────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: Runner Refactoring (3 weeks, after context complete)    │
├──────────────────────────────────────────────────────────────────┤
│ Week 4-5: Decompose PipelineRunner into 6 components             │
│          - PipelineOrchestrator                                  │
│          - PipelineExecutor                                      │
│          - StepRunner                                            │
│          - StepParser                                            │
│          - ControllerRouter                                      │
│          - ArtifactManager                                       │
│ Week 6: Integration testing and examples validation              │
└──────────────────────────────────────────────────────────────────┘
```

---

## Dependency Analysis

### Why Context Must Come First

**The Problem:**
- PipelineRunner refactoring assumes typed context
- Cannot decompose runner without clear context interface
- Controllers need explicit communication protocol
- Dataset operations tied to context structure

**The Evidence:**
```python
# Current runner code (CANNOT refactor without typed context)
context = {"partition": "train", "processing": [["raw"]]}  # What is this?
context, artifacts = controller.execute(step, op, dataset, context, runner)
X = dataset.x(context, layout="2d")  # What does context contain?
```

**After Context Refactoring:**
```python
# Clear interfaces enable runner decomposition
context = ExecutionContext(
    selector=DataSelector(partition="train", processing=[["raw"]]),
    state=PipelineState(y_processing="numeric"),
    metadata=StepMetadata(keyword="transform")
)
context, artifacts = controller.execute(step, op, dataset, context, runner)
X = dataset.x(context.get_selector(), layout="2d")  # Type-safe
```

### Component Dependencies

```
ExecutionContext (context.py)
    ↓
DataSelector → dataset.x() / dataset.y()
    ↓
PipelineState → Controllers (transformers, models)
    ↓
StepMetadata → ControllerRouter (matching logic)
    ↓
PipelineExecutor → StepRunner → Controllers
    ↓
PipelineOrchestrator → Workspace
```

---

## Detailed Timeline

### Weeks 1-3: Context Foundation (CRITICAL)

**Week 1: New Context Module**
- [ ] Create `nirs4all/pipeline/context.py`
- [ ] Implement DataSelector (with processing for future caching)
- [ ] Implement PipelineState (mode, y_processing, step tracking)
- [ ] Implement StepMetadata (controller coordination flags)
- [ ] Implement ExecutionContext (with custom dict for extensibility)
- [ ] Write 50+ unit tests
- [ ] Documentation
- **Deliverable**: Context classes with 100% test coverage

**Week 2: Atomic Migration**
- [ ] Update `dataset.py` Selector type to DataSelector
- [ ] Update `dataset.x()` and `dataset.y()` signatures
- [ ] Update `runner.py` to create ExecutionContext
- [ ] Update ALL 18+ controllers to use typed context
- [ ] Update ALL tests to use typed context
- [ ] Use type checker to find all Dict[str, Any] context usage
- [ ] Fix all 176+ usage locations
- **Deliverable**: Fully migrated codebase, type checker passes

**Week 3: Validation & Performance**
- [ ] All unit tests pass
- [ ] Run `.\run.ps1 -l` in examples/ (integration tests)
- [ ] Performance benchmarks (<10% overhead acceptable)
- [ ] Type checker validation (mypy/pyright)
- [ ] Documentation updates
- **Deliverable**: Production-ready typed context

### Weeks 4-6: Runner Decomposition (Unblocked)

**Week 4: Component Extraction**
- [ ] Create `nirs4all/pipeline/execution/` package
- [ ] Extract StepParser from runner.py
- [ ] Extract ControllerRouter from runner.py
- [ ] Extract ArtifactManager from runner.py
- [ ] Unit tests for each component
- **Deliverable**: Three independent components

**Week 5: Executor Separation**
- [ ] Create PipelineExecutor (single pipeline execution)
- [ ] Create StepRunner (single step execution)
- [ ] Refactor runner.py to PipelineOrchestrator
- [ ] Update imports throughout codebase
- **Deliverable**: Six-component architecture

**Week 6: Integration & Testing**
- [ ] Integration tests for full pipeline
- [ ] Performance benchmarks (compare with old runner)
- [ ] Regression tests (all examples must pass)
- [ ] Documentation updates
- **Deliverable**: Production-ready refactored runner

**Total Timeline**: 6 weeks for complete refactoring

---

## Risk Assessment

### High Risk (Mitigation Required)

1. **Atomic context migration breaks everything**
   - **Mitigation**: Use type checker to guide migration systematically
   - **Testing**: Fix compile errors first, then runtime errors
   - **Rollback**: Git branch for atomic commit, can revert cleanly
   - **Acceptance**: Breaking changes are ALLOWED per user requirements

2. **Runner refactoring introduces regressions**
   - **Mitigation**: Run all examples after each phase
   - **Testing**: `.\run.ps1 -l` in examples/ (integration tests)
   - **Rollback**: Keep old runner structure in separate branch

3. **Performance degradation from context overhead**
   - **Mitigation**: Benchmark context creation/copy
   - **Testing**: Performance tests comparing old vs new
   - **Optimization**: Profile-guided optimization if needed
   - **Acceptance**: <10% overhead acceptable

### Medium Risk (Monitor)

4. **Custom controller data mechanism insufficient**
   - **Mitigation**: `context.custom` dict provides full flexibility
   - **Testing**: Verify custom controllers can store arbitrary data

5. **Processing chains in selector causes confusion**
   - **Mitigation**: Clear documentation on why (future caching/flow control)
   - **Testing**: Ensure selector and state both work correctly

### Low Risk

6. **Documentation outdated**
   - **Mitigation**: Update docs in same PR as code changes

**Key Change**: NO backward compatibility means lower risk of maintaining two code paths simultaneously

---

## Success Metrics

### Context Refactoring Success

- [ ] 176+ context usage locations migrated atomically
- [ ] Type checker passes (mypy/pyright)
- [ ] All existing tests pass (with updated signatures)
- [ ] Type hints throughout context code
- [ ] <10% performance overhead
- [ ] Custom controller extensibility verified

### Runner Refactoring Success

- [ ] PipelineRunner reduced from 1050 to <200 lines
- [ ] 6 components with single responsibilities
- [ ] All components <300 lines
- [ ] 90%+ test coverage on new components
- [ ] All examples pass (`.\run.ps1 -l` clean)
- [ ] <5% performance degradation

### Overall Success

- [ ] Code maintainability score improved
- [ ] New contributors can add controllers easily
- [ ] New operator types don't require runner changes
- [ ] Context modifications have clear typed interface
- [ ] Custom controllers can propagate data via context.custom
- [ ] Documentation complete and accurate

---

## Rollback Plan

### If Context Refactoring Fails

1. Revert atomic migration commit
2. Keep new context module as experimental
3. Analyze failures and retry with fixes

**Note**: No backward compatibility to maintain means clean revert possible

### If Runner Refactoring Fails

1. Revert runner decomposition
2. Keep new components as experimental alternatives
3. Gradual adoption by new features only

---

## Resolved Design Questions

**Processing Chains Location**:
- ✅ Stay in DataSelector (not PipelineState)
- Rationale: Future flow controllers with multiple processing paths
- Rationale: Feature caching requires processing for data selection

**Backward Compatibility**:
- ✅ NO backward compatibility
- Rationale: Faster implementation, no technical debt
- Rationale: Only public API signatures need attention
- Rationale: Internal refactoring can break freely

**Custom Controller Data**:
- ✅ Use `context.custom` dict
- Rationale: Full flexibility for controller-specific data
- Rationale: Doesn't pollute core context schema

**fold_id Handling**:
- ✅ Optional field in DataSelector
- Rationale: Used by CV operations if exists, ignored otherwise

---

## Communication Plan

### Internal Team

- **Kickoff Meeting**: Review proposals, agree on timeline
- **Daily Standups**: Progress tracking during atomic migration (Week 2)
- **Code Reviews**: All PRs reviewed by 2+ developers
- **Testing Sessions**: After each week, run full test suite

### External Users (if applicable)

- **Announcement**: Breaking changes coming in next release
- **Migration Guide**: How to update custom controllers
- **Release Notes**: Clear documentation of breaking changes
- **Timeline**: Give advance notice before release

**Note**: No deprecation period needed per user requirements

---

## Next Steps

### Immediate (This Week)

1. [ ] Review all three refactoring proposals
2. [ ] Approve context refactoring as blocker
3. [ ] Assign developers to context Phase 1
4. [ ] Create GitHub issues for each phase
5. [ ] Set up project board for tracking

### Short Term (Weeks 1-3)

1. [ ] Execute context refactoring atomically
2. [ ] Daily progress updates during atomic migration week
3. [ ] Continuous testing with type checker
4. [ ] Documentation updates

### Medium Term (Weeks 4-6)

1. [ ] Runner refactoring execution
2. [ ] Integration testing
3. [ ] Performance validation
4. [ ] Final documentation

---

## Conclusion

This refactoring is **ambitious but necessary** for long-term maintainability. The clean break approach eliminates technical debt.

**Critical Success Factors**:
1. ✅ Context refactoring with clean break (3 weeks)
2. ✅ Type checker guides atomic migration
3. ✅ All tests pass after migration
4. ✅ Examples run cleanly (`.\run.ps1 -l`)
5. ✅ Performance within 10% of current implementation
6. ✅ Custom controller extensibility via context.custom
7. ✅ Processing chains in DataSelector for future caching

**Estimated Total Effort**: 6 weeks with 2 developers, 9 weeks with 1 developer

**Risk Level**: Medium-High (atomic migration is all-or-nothing, but type checker helps)

**Reward**: Maintainable, extensible, type-safe pipeline infrastructure with no technical debt

**Key Advantages of Clean Break**:
- 6 weeks total vs 8+ weeks with backward compatibility
- No compatibility layer maintenance burden
- Clear before/after boundary
- Type safety from day one
- No gradual migration complexity

---

**Status**: Awaiting approval to begin Phase 1
