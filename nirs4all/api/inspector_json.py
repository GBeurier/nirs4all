"""JSON-safe values shared by result views and the isolated document host."""

import math
from datetime import date, datetime, time
from typing import Any


def sanitize_float(value: Any) -> Any:
    """Convert NaN / Inf floats to ``None``; pass every other value through."""
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return sanitize_dict(value)
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    return sanitize_float(value)


def sanitize_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Preserve timestamps and recursively encode non-finite values as null."""
    return {key: _sanitize_value(value) for key, value in d.items()}
