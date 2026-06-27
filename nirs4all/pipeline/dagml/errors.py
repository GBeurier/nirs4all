"""Failure types and fail-loud guards for the dag-ml host backend.

``DagMlUnsupported`` is the catchable error every unsupported-or-failed dag-ml shape raises;
``_cli_child_error`` extracts the real cause from a dag-ml-cli failure; ``_reject_multi_model``
rejects a multi-model pipeline UP FRONT.
"""

from __future__ import annotations

from typing import Any


class DagMlUnsupported(NotImplementedError):
    """A pipeline shape (or a dag-ml-cli run of one) the dag-ml backend cannot execute.

    A subclass of ``NotImplementedError`` so the planned cutover fallback
    (``try dag-ml; except NotImplementedError → legacy``) catches it: an unsupported-or-failed
    dag-ml shape must NEVER escape ``run_via_dagml`` as a bare ``RuntimeError``/``ValueError``
    (which the fallback would not catch and would hard-fail in production). Raised both for shapes
    rejected UP FRONT (no splitter, multiple model nodes, …) and for a dag-ml-cli run that failed
    mid-execution — in either case the fallback redirects the pipeline to the legacy engine.
    """


def _cli_child_error(stdout: str) -> str:
    """The child adapter's actual error line(s) from a dag-ml-cli failure, for an informative message.

    The captured ``stdout``+``stderr`` ends with the process-adapter traceback; the last
    ``Error:``/``ValueError:``/``…Error:`` line is the real cause (e.g. ``ValueError: data view is
    missing``). Surfacing it beats the bare ``rc=1`` (the legacy E-QUALITY message names the cause),
    and lets the reader see WHY the shape failed. Falls back to the trailing slice when no error line
    is found, so nothing is ever hidden.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    causes = [line for line in lines if line.startswith("Error:") or ("Error:" in line and not line.startswith("File "))]
    return causes[-1] if causes else stdout[-800:]


def _reject_multi_model(steps: list[Any]) -> None:
    """Reject a concrete pipeline carrying MORE THAN ONE ``{"model": ...}`` step (fail loud UP FRONT).

    The dag-ml CV+refit assembly (:func:`assemble_cv_refit_dsl` → ``model_node_id``) binds the data
    source to exactly ONE model node; several top-level model steps compile to several model nodes,
    only the first of which gets a data binding, so the others reach the runtime with an empty data
    view and crash mid-execution inside the adapter (``node_runner._sample_ids`` → ``data view is
    missing`` → a bare ``rc=1``). Detecting it here turns that into a CATCHABLE up-front error.

    Several models in one nirs4all pipeline (e.g. ``U02``'s three ``{"model": PLS(n)}`` steps) is a
    legacy multi-model run — the legacy engine fans them out into separate chains; the dag-ml backend
    runs ONE model per pipeline (a model sweep uses ``{"model": {"_or_": [...]}}``, which expands to
    one model PER variant). Until per-model fan-out is wired, this shape falls back to legacy.
    """
    model_steps = [step for step in steps if isinstance(step, dict) and "model" in step]
    if len(model_steps) > 1:
        raise DagMlUnsupported(
            f"engine='dag-ml' runs ONE model per pipeline, but this pipeline has {len(model_steps)} "
            "top-level {'model': ...} steps; the dag-ml CV+refit binds data to a single model node, so "
            "multiple model nodes crash mid-run (data view is missing). Use a model sweep "
            "({'model': {'_or_': [...]}}) — which runs one model per variant — or the legacy engine."
        )
