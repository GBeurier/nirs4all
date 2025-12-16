# ⚠️ SUPERSEDED BY V3 DESIGN

> **This document is obsolete.** See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) for the current artifact system design and implementation roadmap.
>
> This document is preserved for historical reference only. It tracks progress on the V2 implementation which has been superseded.

---

# Artifacts System V2 Roadmap (ARCHIVED)

**Status**: Superseded by V3 (December 2025)
**Reason**: V2 implementation revealed limitations that require architectural changes. See [ARTIFACT_SYSTEM_CURRENT_STATE.md](./ARTIFACT_SYSTEM_CURRENT_STATE.md) for the analysis.

---

## Summary of V2 Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Foundation | ✅ Complete | Types, registry, manifest schema |
| Phase 2: Loader | ✅ Complete | Artifact loading with deduplication |
| Phase 3: Controllers | ⚠️ Partial | Works for simple cases, fails for branches |
| Phase 4: Stacking | ⚠️ Partial | Training works, reload has issues |
| Phase 5: Cleanup | ✅ Complete | Orphan detection, CLI tools |
| Phase 6: Remote | 🚫 Deferred | Out of scope |

## V2 Remaining Issues (Addressed in V3)

1. **Branch substeps not recorded in trace** → V3 adds substep recording
2. **Operation counter diverges** → V3 uses chain-based identification
3. **Multi-source + branch fails** → V3 includes source_index in chain
4. **Nested branches unsupported** → V3 handles arbitrary nesting

---

## V3 Implementation Roadmap

See [ARTIFACT_SYSTEM_V3_DESIGN.md](./ARTIFACT_SYSTEM_V3_DESIGN.md) Section 13 for the new implementation phases.
