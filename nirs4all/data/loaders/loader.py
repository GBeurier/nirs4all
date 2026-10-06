# dataset_loader.py

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union, cast

import numpy as np
import pandas as pd

from nirs4all.data.config_parser import parse_config

# Import the new loader system
from nirs4all.data.loaders.base import FormatNotSupportedError, LoaderRegistry, apply_na_policy
from nirs4all.data.loaders.csv_loader_new import load_csv
from nirs4all.data.signal_type import SignalType, normalize_signal_type

if TYPE_CHECKING:
    from nirs4all.data.relations import NormalizedObservationTable


def _merge_params(local_params, handler_params, global_params):
    """
    Merge parameters from local, handler, and global scopes.

    Parameters:
    - local_params (dict): Local parameters specific to the data subset.
    - handler_params (dict): Parameters specific to the handler.
    - global_params (dict): Global parameters that apply to all handlers.

    Returns:
    - dict: Merged parameters with precedence: local > handler > global.
    """
    merged_params = {} if global_params is None else global_params.copy()
    if handler_params is not None:
        merged_params.update(handler_params)
    if local_params is not None:
        merged_params.update(local_params)
    return merged_params

# Known loading parameter keys that can appear at root level of config
_LOADING_PARAM_KEYS = frozenset({
    'delimiter', 'decimal_separator', 'has_header',
    'na_policy', 'na_fill_config', 'header_unit', 'categorical_mode', 'signal_type',
    'encoding', 'keep_default_na', 'na_values', 'na_filter'
})

def _get_effective_global_params(config: dict[str, Any]) -> dict[str, Any] | None:
    """
    Get effective global params by merging root-level loading params with global_params.

    Root-level params have lowest precedence (global_params overrides them).
    This allows users to write simpler configs like:
        {"x_train": "path.csv", "delimiter": ",", "has_header": True}
    instead of:
        {"x_train": "path.csv", "global_params": {"delimiter": ",", "has_header": True}}

    Parameters:
    - config (dict): The full configuration dictionary.

    Returns:
    - dict or None: Merged params with precedence: global_params > root-level params.
    """
    # Extract known loading params from root level
    root_params = {k: v for k, v in config.items() if k in _LOADING_PARAM_KEYS}

    global_params = config.get('global_params')

    if not root_params and not global_params:
        return None

    # Merge: root_params (lowest) < global_params (higher)
    effective = root_params.copy() if root_params else {}
    if global_params:
        effective.update(global_params)

    return effective if effective else None

def _load_file_with_registry(
    file_path: str | Path,
    header_unit: str = "cm-1",
    data_type: str = "x",
    **params: Any,
) -> tuple[pd.DataFrame | None, dict[str, Any], pd.Series | None, list[str], str]:
    """Load a file using the LoaderRegistry for format detection.

    This function provides automatic format detection and loading using
    the registered file loaders. It falls back to CSV loading for unknown formats.

    Args:
        file_path: Path to the file to load.
        header_unit: Unit for headers ('cm-1', 'nm', etc.).
        data_type: Type of data ('x', 'y', or 'metadata').
        **params: Additional loading parameters.

    Returns:
        Tuple of (DataFrame, report, na_mask, headers, header_unit).
    """
    if isinstance(file_path, (np.ndarray, pd.DataFrame, pd.Series)):
        # In-memory and disk inputs share positional alignment. Preserve the
        # historical array NA default; explicitly requested policies still apply.
        if isinstance(file_path, np.ndarray):
            if file_path.ndim not in (1, 2):
                raise ValueError("In-memory input must be one- or two-dimensional")
            frame = pd.DataFrame(file_path.reshape(-1, 1) if file_path.ndim == 1 else file_path)
            prefix = "feature" if data_type == "x" else "meta" if data_type == "metadata" else "target"
            frame.columns = [f"{prefix}_{i}" for i in range(frame.shape[1])]
        else:
            frame = pd.DataFrame(file_path).copy()
            frame.columns = frame.columns.astype(str)
        frame = frame.reset_index(drop=True)
        initial_shape = frame.shape
        na_mask = frame.isna().any(axis=1)
        frame, na_report = apply_na_policy(frame, params.get("na_policy", "ignore"), params.get("na_fill_config"))
        return frame, {"initial_shape": initial_shape, "na_handling": na_report}, na_mask, frame.columns.tolist(), header_unit

    path = Path(file_path) if isinstance(file_path, str) else file_path

    # Try to use the registry for format detection
    try:
        registry = LoaderRegistry.get_instance()
        loader = registry.get_loader(path)
        if path.suffix.lower() in {".parquet", ".pq"}:
            params.pop("encoding", None)
        result = loader.load(
            path,
            header_unit=header_unit,
            data_type=data_type,
            **params,
        )
        # This entrypoint aligns files positionally. Stored DataFrame index
        # labels (e.g. in Parquet) must not accidentally become join keys.
        if result.data is not None:
            if result.report.get("na_handling", {}).get("strategy") == "remove_sample" and result.na_mask is not None:
                result.data = result.data.copy()
                result.data.index = np.flatnonzero(~result.na_mask.to_numpy(dtype=bool))
            else:
                result.data = result.data.reset_index(drop=True)
        return (
            result.data,
            result.report,
            result.na_mask,
            result.headers,
            result.header_unit,
        )
    except FormatNotSupportedError:
        # Fall back to the unified CSV loader for unknown formats
        return cast(tuple[pd.DataFrame | None, dict[str, Any], pd.Series | None, list[str], str], load_csv(file_path, header_unit=header_unit, data_type=data_type, **params))
    except Exception as e:
        # On any other error, try CSV as a fallback
        try:
            return cast(tuple[pd.DataFrame | None, dict[str, Any], pd.Series | None, list[str], str], load_csv(file_path, header_unit=header_unit, data_type=data_type, **params))
        except Exception:
            # If CSV also fails, re-raise the original error
            raise e from None

def load_XY(x_path: str, x_filter: Any, x_params: dict[str, Any], y_path: str | None, y_filter: Any, y_params: dict[str, Any], m_path: str | None = None, m_filter: Any = None, m_params: dict[str, Any] | None = None, *, row_info: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray, Any, list[str], list[str], str, SignalType | None]:
    """
    Load X, Y, and metadata from single paths. For multi-source, this will be called multiple times.

    Parameters:
    - x_path (str): Single path to X data file.
    - x_filter: Filter to apply to X data (not implemented yet).
    - x_params (dict): Parameters for loading X data, including:
        - header_unit: Unit for headers ("cm-1", "nm", "none", "text", "index")
        - signal_type: Signal type ("absorbance", "reflectance", "reflectance%", etc.)
        - delimiter, decimal_separator, has_header, na_policy, etc.
    - y_path (str): Path to the Y data file (can be None).
    - y_filter: Filter to apply to Y data (or indices if y_path is None).
    - y_params (dict): Parameters for loading Y data.
    - m_path (str): Path to metadata file (can be None).
    - m_filter: Filter to apply to metadata (not implemented yet).
    - m_params (dict): Parameters for loading metadata.
    - row_info (dict): Optional output for original row count and retained indices.

    Returns:
    - tuple: (x, y, m, x_headers, m_headers, x_header_unit, x_signal_type) where:
        - x, y, m are numpy arrays/DataFrames
        - x_headers, m_headers are lists of column names
        - x_header_unit is the unit string for X headers ("cm-1", "nm", "none", "text", "index")
        - x_signal_type is the signal type (SignalType enum or None for auto-detect)

    Raises:
    - ValueError: If data is invalid or if there are inconsistencies.
    """
    if x_path is None:
        raise ValueError("Invalid x definition: x_path is None")

    x_params = x_params.copy()
    # Set default parameters
    if 'categorical_mode' not in x_params:
        x_params['categorical_mode'] = 'auto'
    if 'data_type' not in x_params:
        x_params['data_type'] = 'x'

    # Extract header_unit from params (default to cm-1)
    x_header_unit = x_params.pop('header_unit', 'cm-1')

    # Extract signal_type from params (default to None for auto-detect)
    x_signal_type_raw = x_params.pop('signal_type', None)
    x_signal_type: SignalType | None = None
    if x_signal_type_raw is not None:
        x_signal_type = normalize_signal_type(x_signal_type_raw)

    # Load X data using format-aware loader
    try:
        x_df, x_report, x_na_mask, x_headers, x_unit = _load_file_with_registry(
            x_path, header_unit=x_header_unit, **x_params
        )
        if x_report.get("error") is not None or x_df is None:
            raise ValueError(f"Failed to load X data from {x_path}: {x_report.get('error', 'Unknown error')}")
    except Exception as e:
        raise ValueError(f"Error loading X data from {x_path}: {str(e)}") from e

    if x_filter is not None:
        raise NotImplementedError("Auto-filtering not implemented yet")

    initial_rows = x_report["initial_shape"][0]

    # Load Y data
    if y_path is None and y_filter is None:
        # No Y data to extract - create empty Y array with same number of rows as X
        y_df = pd.DataFrame(index=x_df.index)  # Empty DataFrame with matching index
    elif y_path is None:
        # Y is a subset of X
        if not all(isinstance(i, int) for i in y_filter):
            raise ValueError("Invalid y definition: y_filter is not a list of integers")

        if x_df.shape[1] <= max(y_filter):
            raise ValueError(f"Y filter indices {y_filter} exceed X columns ({x_df.shape[1]})")

        # Extract Y from X and remove Y columns from X
        y_df = x_df.iloc[:, y_filter]
        x_df = x_df.drop(x_df.columns[y_filter], axis=1)
    else:
        # Y is in a separate file
        try:
            y_params_copy = y_params.copy()
            if 'categorical_mode' not in y_params_copy:
                y_params_copy['categorical_mode'] = 'auto'
            if 'data_type' not in y_params_copy:
                y_params_copy['data_type'] = 'y'

            y_df, y_report, y_na_mask, _, _ = _load_file_with_registry(y_path, **y_params_copy)
            if y_report.get("error") is not None or y_df is None:
                raise ValueError(f"Failed to load Y data from {y_path}: {y_report.get('error', 'Unknown error')}")
            if initial_rows != y_report["initial_shape"][0]:
                raise ValueError(f"Row count mismatch: X({initial_rows}) Y({y_report['initial_shape'][0]})")
            # Targets fits one encoder on original training labels and reuses
            # it for held-out labels, instead of retaining file-local codes.
            for column, info in y_report.get("categorical_info", {}).items():
                if column in y_df:
                    y_df[column] = y_df[column].map(dict(enumerate(info["categories"])))
        except Exception as e:
            raise ValueError(f"Error loading Y data from {y_path}: {str(e)}") from e

        if y_filter is not None:
            if not all(isinstance(i, int) for i in y_filter):
                raise ValueError("Invalid y_filter: must be list of integers")
            if y_df.shape[1] <= max(y_filter):
                raise ValueError(f"Y filter indices {y_filter} exceed Y columns ({y_df.shape[1]})")
            y_df = y_df.iloc[:, y_filter]

    # Load metadata if provided
    m_df = pd.DataFrame()
    m_headers: list[str] = []
    if m_path is not None:
        try:
            if m_params is None:
                m_params = {}
            m_params_copy = m_params.copy()
            if 'categorical_mode' not in m_params_copy:
                m_params_copy['categorical_mode'] = 'preserve'  # Keep original types for metadata
            if 'data_type' not in m_params_copy:
                m_params_copy['data_type'] = 'metadata'
            # Metadata permits missing values by default, but participates in
            # joint row removal when that policy is explicitly requested.
            if m_params_copy.get('na_policy') != 'remove_sample':
                m_params_copy['na_policy'] = 'ignore'

            m_df, m_report, m_na_mask, m_headers, _ = _load_file_with_registry(m_path, **m_params_copy)

            if m_report.get("error") is not None or m_df is None:
                raise ValueError(f"Failed to load metadata from {m_path}: {m_report.get('error', 'Unknown error')}")
        except Exception as e:
            raise ValueError(f"Error loading metadata from {m_path}: {str(e)}") from e

        if m_filter is not None:
            raise NotImplementedError("Metadata filtering not implemented yet")

        if initial_rows != m_report["initial_shape"][0]:
            raise ValueError(f"Row count mismatch: X({initial_rows}) Metadata({m_report['initial_shape'][0]})")

    # Retain original row identity after independent NA removal. Pair only
    # rows present in every frame, in the original spectrum order.
    common_rows = x_df.index.intersection(y_df.index, sort=False)
    if m_path is not None:
        common_rows = common_rows.intersection(m_df.index, sort=False)
        m_df = m_df.loc[common_rows]
    x_df = x_df.loc[common_rows]
    y_df = y_df.loc[common_rows]
    if row_info is not None:
        row_info.update(initial_rows=initial_rows, indices=common_rows)

    # Update x_headers after potential column removal (if Y was extracted from X)
    x_headers = x_df.columns.tolist()

    # Convert to numpy arrays
    try:
        x = x_df.astype(np.float32).values if not x_df.empty else np.empty((0, x_df.shape[1]), dtype=np.float32)
        y = y_df.values
        # Keep metadata as DataFrame (don't convert to numeric)
        m = m_df if not m_df.empty else None
    except Exception as e:
        raise ValueError(f"Error converting data to numpy arrays: {str(e)}") from e

    return x, y, m, x_headers, m_headers, x_unit, x_signal_type

def _audit_multisource_lengths(config: dict[str, Any], x_arrays: list[np.ndarray], *, original_lengths: list[int] | None = None) -> None:
    """Reject heterogeneous multi-source feature blocks loaded positionally.

    Compares the row count of every loaded source. Equal counts pass through
    (the legacy aligned case is unchanged). Unequal counts raise an actionable
    :class:`~nirs4all.data.relations.RelationValidationError`, tailored by
    whether a ``link_by`` key is declared on the sources.

    Args:
        config: The dataset configuration dict (carries ``_sources`` / link_by).
        x_arrays: The per-source loaded feature arrays.
        original_lengths: Row counts before NA removal, when available.
    """
    from nirs4all.data.relations import audit_source_lengths, parse_relation_config

    lengths = original_lengths if original_lengths is not None else [int(arr.shape[0]) for arr in x_arrays if hasattr(arr, "shape") and arr.ndim >= 1]
    if len(lengths) <= 1:
        return

    relation_config = parse_relation_config(config)
    link_by = relation_config.link_by if relation_config is not None else None

    # The legacy loader can only align positionally; the relation table that owns
    # heterogeneous alignment is materialised via ``materialize_relation_table`` /
    # ``RawMultiSourceDataset`` (phase N2/N3), so relation_mode stays False here and
    # unequal lengths always fail loudly with an actionable, link_by-aware hint.
    audit_source_lengths(lengths, relation_mode=False, link_by=link_by)


def materialize_relation_table(
    config: dict[str, Any],
    source_frames: dict[str, Any],
    *,
    partition: str = "train",
    target_col: str | None = None,
) -> "NormalizedObservationTable | None":
    """Materialise a :class:`NormalizedObservationTable` from per-source frames.

    This is the loader-level seam for the experimental relation pipeline
    (roadmap N2): given the per-source column frames already read from disk (or
    any ``frame[column]`` container -- pandas ``DataFrame``, dict of lists ...)
    and a declared ``repetition_spec`` / ``link_by``, it joins the sources *by
    key* into a validated relation table. Source row order is irrelevant -- only
    the ``link_by`` keys matter -- so shuffled sources are supported and
    heterogeneous sources are never aligned by accident.

    This function is intentionally **additive**: it does not change the legacy
    :func:`handle_data` return contract. Callers in the relation profile invoke
    it explicitly; legacy positional loads never reach it.

    Args:
        config: The dataset configuration dict (carries the relational fields).
        source_frames: Mapping ``source_id -> frame`` where ``frame[column]``
            yields a per-row sequence. Each source must expose the join-key
            column declared by the spec.
        partition: Partition label applied to every produced row.
        target_col: Optional column holding the per-row (sample-level) target,
            shared across sources via the join.

    Returns:
        A validated :class:`~nirs4all.data.relations.NormalizedObservationTable`,
        or ``None`` when no executable ``repetition_spec`` is declared.

    Raises:
        RelationValidationError: If the declared join is not executable (missing
            key column, divergent / non-unique ids, missing source ...).
    """
    from nirs4all.data.relations import (
        SourceObservations,
        build_relation_table,
        parse_relation_config,
    )

    relation_config = parse_relation_config(config)
    if relation_config is None or relation_config.spec is None:
        return None

    spec = relation_config.spec
    sources = []
    for source_id in sorted(source_frames):
        frame = source_frames[source_id]
        source_spec = spec.source_spec(source_id)
        # Targets are sample-level: a `link_by`-shared target may live on a single
        # source. Only extract it where the column actually exists; the join
        # propagates it to the physical sample (and thus its other sources).
        effective_target = target_col if (target_col is not None and target_col in frame) else None
        sources.append(
            SourceObservations.from_frame(
                source_id,
                frame,
                sample_col=spec.join_key,
                rep_col=source_spec.rep_col,
                target_col=effective_target,
            )
        )
    return build_relation_table(spec, sources, partition=partition)


def _reject_unloadable_relation_config(config: dict[str, Any]) -> None:
    """Refuse experimental source-aware relation configs in the legacy loader.

    The legacy positional loader concatenates per-source feature blocks by row
    position. It cannot execute the experimental source-aware relation pipeline
    (``experimental_relation_pipeline`` / ``repetition_spec`` declared together
    with multiple ``sources``): such a config exposes no top-level feature matrix,
    so the legacy path would otherwise yield a silently empty dataset (file
    configs, which keep ``sources``) or an opaque positional-alignment error
    (dict configs, which expand to a ``train_x`` list). Heterogeneous sources are
    joined by key and materialised through :class:`RawMultiSourceDataset` and the
    ``rep_fusion`` pipeline step, never through :class:`DatasetConfigs`.

    Args:
        config: The (legacy-shaped) dataset configuration dict.

    Raises:
        RelationValidationError: If the config opts into the relation pipeline
            (enabled flag or a repetition spec) and declares multiple sources.
    """
    from nirs4all.data.relations import RelationValidationError, parse_relation_config

    relation_config = parse_relation_config(config)
    if relation_config is None or not (relation_config.enabled or relation_config.spec is not None):
        return
    declares_sources = (
        (isinstance(config.get("sources"), list) and len(config["sources"]) > 0)
        or bool(config.get("_sources"))
        or isinstance(config.get("train_x"), list)
        or isinstance(config.get("test_x"), list)
    )
    if not declares_sources:
        return
    raise RelationValidationError(
        "This dataset opts into the experimental source-aware relation pipeline "
        "(experimental_relation_pipeline / repetition_spec) with multiple 'sources', which the legacy "
        "loader cannot materialise: it would align heterogeneous sources by row position or load nothing "
        "at all. Build the relation staging (RawMultiSourceDataset) and materialise a declared "
        "representation with the 'rep_fusion' pipeline step instead of loading this schema through "
        "DatasetConfigs.",
        code="REL-E023",
    )



def _reject_unsupported_positional_options(config: dict[str, Any]) -> None:
    """Reject declarations this loader cannot execute, before caching or mutation.

    Config folds are not materialized here; callers may set validated folds on
    a loaded SpectroDataset, or use a pipeline splitter/FoldFileLoader.
    Key joins remain available through the explicit relation materialization API.
    """
    if config.get("folds") is not None:
        raise ValueError("Dataset-config folds are not supported by the positional loader. Use dataset.set_folds() after loading, or a pipeline splitter/FoldFileLoader.")
    for key in ("shared_targets", "shared_metadata", "targets", "metadata"):
        definitions = config.get(key)
        if not definitions:
            continue
        definitions = definitions if isinstance(definitions, list) else [definitions]
        partitions: set[str] = set()
        for definition in definitions:
            if not isinstance(definition, dict):
                continue
            for option in ("columns", "rows", "link_by"):
                if definition.get(option) is not None:
                    raise ValueError(f"{key}.{option} is not supported by the positional dataset loader. Use explicit inputs or the relation materialization API.")
            for partition in ([definition["partition"]] if definition.get("partition") else ["train", "test"]):
                if partition in partitions:
                    raise ValueError(f"Multiple {key} files for partition '{partition}' are not supported by the positional dataset loader")
                partitions.add(partition)
    for key in ("_sources", "sources", "_variations", "variations", "files"):
        for definition in config.get(key) or []:
            if not isinstance(definition, dict):
                continue
            for item in [definition, *(definition.get("files") or [])]:
                if isinstance(item, dict):
                    for option in ("columns", "rows", "link_by"):
                        if item.get(option) is not None:
                            raise ValueError(f"{key}.{option} is not supported by the positional dataset loader. Use the relation materialization API for key joins.")

def handle_data(config, t_set):
    """
    Handle data loading for a given dataset type (train, test).
    Supports both single-source and multi-source datasets.

    Parameters:
    - config (dict): Data configuration dictionary.
    - t_set (str): The dataset type ('train', 'test').

    Returns:
    - tuple: (x, y, m, x_headers, m_headers, x_header_unit, x_signal_type) where:
        - x is numpy array or list of arrays
        - y is numpy array
        - m is DataFrame or None (metadata)
        - x_headers is list of column names or list of lists for multi-source
        - m_headers is list of metadata column names
        - x_header_unit is string or list of strings for multi-source ("cm-1", "nm", "none", "text", "index")
        - x_signal_type is SignalType or list of SignalType for multi-source (None for auto-detect)
    """
    if config is None:
        raise ValueError(f"Configuration for {t_set} dataset is None")

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config type for {t_set}: {type(config)}")

    # Experimental source-aware relation configs are not loadable by the legacy
    # positional loader; fail loudly instead of returning an empty dataset.
    _reject_unloadable_relation_config(config)
    _reject_unsupported_positional_options(config)

    # Get effective global params (includes root-level loading params)
    effective_global_params = _get_effective_global_params(config)

    # Get paths
    x_path = config.get(f'{t_set}_x')
    y_path = config.get(f'{t_set}_y')
    m_path = config.get(f'{t_set}_group')  # Metadata uses 'group' key

    x_filter = config.get(f'{t_set}_x_filter')
    y_filter = config.get(f'{t_set}_y_filter')
    m_filter = config.get(f'{t_set}_group_filter')

    # Handle multi-source X data
    if isinstance(x_path, list):
        if not x_path:
            raise ValueError("Feature sources must be a nonempty list")
        declared_params = config.get(f'{t_set}_x_params')
        if isinstance(declared_params, list) and len(declared_params) != len(x_path):
            raise ValueError("Per-source parameter count must match feature source count")
        x_arrays = []
        headers_arrays = []
        header_units = []
        signal_types = []
        source_rows: list[dict[str, Any]] = []
        y_array = None
        m_data = None
        m_headers = []

        # Check if we have per-source params
        x_params_config = config.get(f'{t_set}_x_params')

        for i, single_x_path in enumerate(x_path):
            # Determine params for this source
            if isinstance(x_params_config, list) and i < len(x_params_config):
                # Per-source params provided
                source_x_params = _merge_params(x_params_config[i], config.get(f'{t_set}_params'), effective_global_params)
            elif isinstance(x_params_config, dict):
                # Check if dict contains list of units or signal_types for multi-source
                source_params = x_params_config.copy()

                # Handle header_unit list
                if 'header_unit' in x_params_config and isinstance(x_params_config['header_unit'], list):
                    if i < len(x_params_config['header_unit']):
                        source_params['header_unit'] = x_params_config['header_unit'][i]
                    else:
                        source_params['header_unit'] = "cm-1"

                # Handle signal_type list
                if 'signal_type' in x_params_config and isinstance(x_params_config['signal_type'], list):
                    if i < len(x_params_config['signal_type']):
                        source_params['signal_type'] = x_params_config['signal_type'][i]
                    else:
                        source_params['signal_type'] = None

                source_x_params = _merge_params(source_params, config.get(f'{t_set}_params'), effective_global_params)
            else:
                # No params or unsupported format
                source_x_params = _merge_params(None, config.get(f'{t_set}_params'), effective_global_params)

            y_params = _merge_params(config.get(f'{t_set}_y_params'), config.get(f'{t_set}_params'), effective_global_params)
            m_params = _merge_params(config.get(f'{t_set}_group_params'), config.get(f'{t_set}_params'), effective_global_params)

            try:
                row_info: dict[str, Any] = {}
                # For multi-source, only the first source should handle Y and metadata extraction
                if i == 0:
                    x_single, y_array, m_data, x_headers, m_headers, x_unit, x_sig_type = load_XY(
                        single_x_path, x_filter, source_x_params,
                        y_path, y_filter, y_params,
                        m_path, m_filter, m_params, row_info=row_info
                    )
                else:
                    # For additional sources, don't extract Y or metadata
                    x_single, _, _, x_headers, _, x_unit, x_sig_type = load_XY(
                        single_x_path, x_filter, source_x_params,
                        None, None, y_params,
                        None, None, None, row_info=row_info
                    )

                x_arrays.append(x_single)
                headers_arrays.append(x_headers)
                header_units.append(x_unit)
                signal_types.append(x_sig_type)
                source_rows.append(row_info)
            except Exception as e:
                raise ValueError(f"Error loading X source {i} from {single_x_path}: {str(e)}") from e

        # Guardrail (roadmap N0): multi-source feature blocks are concatenated by
        # row position downstream, which only holds when every source has the
        # same number of rows. Heterogeneous repetitions (e.g. MIR=2N/RAMAN=3N)
        # must be joined via a declared relation plan (experimental, phase N3),
        # not loaded by accident as positionally-aligned sources.
        # Original lengths must agree before NA filtering. Then remove the
        # union of missing rows from every source and shared targets/metadata.
        _audit_multisource_lengths(config, x_arrays, original_lengths=[rows["initial_rows"] for rows in source_rows])
        common_rows = source_rows[0]["indices"]
        for rows in source_rows[1:]:
            common_rows = common_rows.intersection(rows["indices"], sort=False)
        for i, rows in enumerate(source_rows):
            positions = rows["indices"].get_indexer(common_rows)
            x_arrays[i] = x_arrays[i][positions]
            if i == 0:
                if y_array is not None:
                    y_array = y_array[positions]
                if m_data is not None:
                    m_data = m_data.iloc[positions]

        return x_arrays, y_array, m_data, headers_arrays, m_headers, header_units, signal_types
    else:
        # Single source
        x_params = _merge_params(config.get(f'{t_set}_x_params'), config.get(f'{t_set}_params'), effective_global_params)
        y_params = _merge_params(config.get(f'{t_set}_y_params'), config.get(f'{t_set}_params'), effective_global_params)
        m_params = _merge_params(config.get(f'{t_set}_group_params'), config.get(f'{t_set}_params'), effective_global_params)
        return load_XY(x_path, x_filter, x_params, y_path, y_filter, y_params, m_path, m_filter, m_params)
