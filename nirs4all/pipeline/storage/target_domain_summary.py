"""Store-v5 provenance for native fold-local target summaries.

These functions copy owner-produced context/scalars. They never pool predictions
or calculate metrics across fitted target domains.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

KEY = "dagml_target_domain_summaries"


class TargetDomainComparisonError(ValueError):
    """Stored results cannot be compared across captured target domains."""


def _document(raw: Any) -> dict[str, Any]:
    value = json.loads(raw) if isinstance(raw, str) else raw
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TargetDomainComparisonError("Stored run summary must be an object")
    return value


def _contexts(summary: dict[str, Any]) -> dict[str, Any]:
    document = summary.get(KEY)
    if document is None:
        return {}
    if (not isinstance(document, dict) or type(document.get("schema_version")) is not int
            or document["schema_version"] != 1 or not isinstance(document.get("chains"), dict)):
        raise TargetDomainComparisonError("Unsupported stored target-domain summary contract")
    return dict(document["chains"])


def merge_contexts(summary: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge chain-keyed metadata without dropping another pipeline's context."""
    merged = dict(_contexts(summary))
    for chain_id, context in incoming.items():
        if (not isinstance(chain_id, str) or not chain_id or not isinstance(context, dict)
                or context.get("cv_summary_fold_id") != "macro"
                or not isinstance(context.get("target_domain_context"), str)
                or not context["target_domain_context"]):
            raise TargetDomainComparisonError("Invalid captured target-domain chain context")
        if chain_id in merged and merged[chain_id] != context:
            raise TargetDomainComparisonError("Cannot replace a captured target-domain chain context")
        merged[chain_id] = dict(context)
    return {**summary, KEY: {"schema_version": 1, "chains": merged}}


def chain_contexts(connection: sqlite3.Connection, chain_ids: list[str]) -> dict[str, dict[str, Any] | None]:
    """Validate macro provenance before any ranking or summary reconstruction."""
    output: dict[str, dict[str, Any] | None] = {}
    unique_ids = list(dict.fromkeys(chain_ids))
    for offset in range(0, len(unique_ids), 400):
        batch = unique_ids[offset:offset + 400]
        markers = ",".join("?" for _ in batch)
        owners = connection.execute(
            "SELECT c.chain_id,r.summary FROM chains c JOIN pipelines p ON p.pipeline_id=c.pipeline_id "
            f"JOIN runs r ON r.run_id=p.run_id WHERE c.chain_id IN ({markers})", batch,
        ).fetchall()
        folds_by_chain: dict[str, set[str]] = {}
        for chain_id, fold_id in connection.execute(
            f"SELECT DISTINCT chain_id,fold_id FROM predictions WHERE chain_id IN ({markers}) AND refit_context IS NULL", batch,
        ):
            folds_by_chain.setdefault(chain_id, set()).add(str(fold_id))
        for chain_id, raw_summary in owners:
            folds = folds_by_chain.get(chain_id, set())
            context = _contexts(_document(raw_summary)).get(chain_id)
            if "macro" not in folds:
                if context is not None:
                    raise TargetDomainComparisonError("Captured macro context has no native macro prediction")
                output[chain_id] = None
                continue
            if folds.intersection({"avg", "w_avg", "fold_avg", "fold_w_avg"}):
                raise TargetDomainComparisonError("Stored chain mixes macrofold and pooled OOF summaries")
            if (not isinstance(context, dict) or context.get("cv_summary_fold_id") != "macro"
                    or not isinstance(context.get("target_domain_context"), str)
                    or not context["target_domain_context"]):
                raise TargetDomainComparisonError(
                    "Stored macro results lack captured target-domain context; inspect the original native run. "
                    "Do not rank or reconstruct their summaries."
                )
            output[chain_id] = dict(context)
    return output


def validate_comparison(contexts: dict[str, dict[str, Any] | None]) -> None:
    """Require the exact SDK context for every candidate before ORDER/LIMIT."""
    if not any(value is not None for value in contexts.values()):
        return
    identities = {value["target_domain_context"] if value is not None else None for value in contexts.values()}
    if None in identities or len(identities) != 1:
        raise TargetDomainComparisonError(
            "Cannot rank results across different native target-domain protocols (macrofold, pooled OOF, "
            "or different bin recipes/partitions). Inspect individual runs or select an explicit source."
        )


def copy_macro_summary(connection: sqlite3.Connection, chain_id: str) -> None:
    """Copy the native score-only report, with exact JSON and no averaging."""
    rows = connection.execute(
        "SELECT val_score,test_score,train_score,scores FROM predictions "
        "WHERE chain_id=? AND fold_id='macro' AND partition='val' AND refit_context IS NULL", [chain_id],
    ).fetchall()
    if len(rows) != 1:
        raise TargetDomainComparisonError("Stored chain requires exactly one native validation macro report")
    macro = rows[0]
    if macro[1] is not None or macro[2] is not None:
        raise TargetDomainComparisonError("A native macro report cannot contain pooled train/test scores")
    count = connection.execute(
        "SELECT COUNT(DISTINCT fold_id) FROM predictions WHERE chain_id=? AND partition='val' "
        "AND refit_context IS NULL AND fold_id NOT IN ('macro','avg','w_avg','fold_avg','fold_w_avg','') "
        "AND SUBSTR(fold_id,-4)!='_agg'", [chain_id],
    ).fetchone()[0]
    if count < 2:
        raise TargetDomainComparisonError("Native macro summary requires at least two real folds")
    connection.execute(
        "UPDATE chains SET cv_val_score=?,cv_test_score=NULL,cv_train_score=NULL,cv_fold_count=?,cv_scores=? "
        "WHERE chain_id=?", [macro[0], count, macro[3], chain_id],
    )
    connection.execute(
        "UPDATE chains SET "
        "model_name=(SELECT model_name FROM predictions WHERE chain_id=? LIMIT 1), "
        "metric=(SELECT metric FROM predictions WHERE chain_id=? LIMIT 1), "
        "task_type=(SELECT task_type FROM predictions WHERE chain_id=? LIMIT 1), "
        "best_params=COALESCE(chains.best_params,(SELECT best_params FROM predictions WHERE chain_id=? "
        "AND best_params IS NOT NULL AND best_params!='{}' LIMIT 1)), "
        "final_test_score=(SELECT test_score FROM predictions WHERE chain_id=? AND fold_id='final' AND partition='test' LIMIT 1), "
        "final_train_score=(SELECT train_score FROM predictions WHERE chain_id=? AND fold_id='final' AND partition='train' LIMIT 1), "
        "final_scores=(SELECT scores FROM predictions WHERE chain_id=? AND fold_id='final' AND partition IN ('test','train') "
        "ORDER BY CASE partition WHEN 'test' THEN 0 ELSE 1 END LIMIT 1) "
        "WHERE chain_id=?", [chain_id] * 8,
    )
