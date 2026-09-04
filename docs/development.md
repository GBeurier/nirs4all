# Developer documentation map

The published `1.0.0` tag is the release baseline. Start with the
[public interface contract](source/reference/public_interfaces.md), the
[V1 migration guide](source/migration/native_v1.md), and the
[developer guide](source/developer/index.md). Source and executable tests take
precedence over historical plans.

Private implementation notes, specifications, reviews and AI handoffs are kept
locally under `docs/_private/`, outside Git and the documentation build. The local
index records their provenance and status. The pre-V1 collection was archived on
2026-09-05 with its topic subdirectories intact; previously tracked originals
remain recoverable from Git at tag `1.0.0`, under `docs/_internal/`.

Use dated, scoped documents with an owner, status and links to source evidence.
Move completed plans into the private archive; keep supported user contracts and
contributor instructions in the public documentation. Do not publish private
datasets, workspaces, credentials, or developer evidence by force-adding it.
