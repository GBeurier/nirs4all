"""Portable, read-only prediction array exports backed by a committed store snapshot."""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import Any

from .workspace_store import WorkspaceStore

MAX_EXPORT_BYTES = 24 * 1024 * 1024
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def export_prediction_arrays(workspace_path: str, request: dict[str, Any]) -> dict[str, str]:
    """Export selected live prediction rows as self-describing Parquet or ZIP.

    Deleted/tombstoned prediction arrays are omitted using the current metadata
    snapshot. Dataset names are resolved through ArrayStore's owner filename
    mapping; sanitized filename collisions remain distinct via dataset_name.
    """
    if set(request) - {"dataset_names", "format", "partition", "model_name"}:
        raise ValueError("Unknown prediction export fields")
    export_format = request.get("format", "zip")
    if export_format not in {"parquet", "zip"}:
        raise ValueError("Prediction export format must be parquet or zip")
    selected = request.get("dataset_names")
    if selected is not None and (not isinstance(selected, list) or not 1 <= len(selected) <= 128 or any(not isinstance(name, str) or not name or len(name) > 256 for name in selected)):
        raise ValueError("Prediction export requires 1 to 128 dataset names")
    for field in ("partition", "model_name"):
        if field in request and (not isinstance(request[field], str) or not request[field] or len(request[field]) > 256):
            raise ValueError(f"Invalid prediction export {field}")
    workspace = Path(workspace_path)
    if not (workspace / "store.sqlite").is_file():
        raise ValueError("not_found: Prediction results store does not exist")
    import polars as pl

    output = io.BytesIO()
    with WorkspaceStore.open_readonly(workspace) as store:
        live = store._fetch_pl("SELECT prediction_id, dataset_name FROM predictions")
        names = sorted(set(live["dataset_name"].to_list()))
        selected = list(dict.fromkeys(selected if selected is not None else names))
        if not selected:
            raise ValueError("not_found: No prediction datasets are available")
        if any(name not in names for name in selected):
            raise ValueError("not_found: Selected prediction dataset does not exist")
        if export_format == "parquet" and len(selected) != 1:
            raise ValueError("Parquet export requires exactly one dataset")
        members: list[tuple[str, bytes]] = []
        seen_paths: set[str] = set()
        total = 0
        for index, name in enumerate(selected):
            source = store.array_store._parquet_path(name)
            if source.is_symlink() or source.parent.is_symlink() or not source.resolve().is_relative_to(workspace.resolve()):
                raise ValueError("Unsafe prediction array file")
            if not source.is_file():
                raise ValueError(f"not_found: Prediction arrays missing for dataset {name}")
            if source.stat().st_size > MAX_SOURCE_BYTES:
                raise ValueError("Prediction array source exceeds 64 MiB")
            frame = pl.read_parquet(source)
            ids = live.filter(pl.col("dataset_name") == name)["prediction_id"].to_list()
            frame = frame.filter(pl.col("prediction_id").is_in(ids) & (pl.col("dataset_name") == name))
            for field in ("partition", "model_name"):
                if field in request:
                    frame = frame.filter(pl.col(field) == request[field])
            buffer = io.BytesIO()
            frame.write_parquet(buffer, compression="zstd")
            data = buffer.getvalue()
            total += len(data)
            if total > MAX_EXPORT_BYTES:
                raise ValueError("Prediction export exceeds 24 MiB")
            member = source.name
            if member in seen_paths:
                member = f"{index}_{member}"
            seen_paths.add(member)
            members.append((member, data))
        if export_format == "parquet":
            filename, content = members[0]
            media_type = "application/octet-stream"
        else:
            with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for member, data in members:
                    archive.writestr(member, data)
            content = output.getvalue()
            filename = "predictions_export.zip"
            media_type = "application/zip"
        if len(content) > MAX_EXPORT_BYTES:
            raise ValueError("Prediction export exceeds 24 MiB")
        return {"filename": filename, "media_type": media_type, "content_base64": base64.b64encode(content).decode("ascii")}
