"""nirs4all → dag-ml(-data) data layer (migration phase 2b-i).

Mechanism-independent pieces shared by both execution paths (CLI process-adapter
and in-process C-ABI): stable sample identity, real-data materialization, the
operator-routing registry, and the data-plan envelope builder. All additive and
import-guarded — production ``nirs4all`` never imports this unless the dag-ml
backend is selected. See ``dag-ml/docs/migration-nirs4all/``.
"""
