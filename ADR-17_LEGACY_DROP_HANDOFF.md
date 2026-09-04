# ADR-17 historical handoff

The June 2026 migration handoff is historical. The current V1 default is `native`;
explicit Python rollback and other engine boundaries are defined by the
[public interface contract](docs/source/reference/public_interfaces.md) and
[native V1 guide](docs/source/migration/native_v1.md).

The full handoff is preserved in the local private archive described in
[developer documentation](docs/development.md), and in Git at tag `1.0.0` under
this original filename. Historical instructions to remove legacy code or commit
local native binaries are not current release instructions.
