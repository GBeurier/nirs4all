"""Shared helpers for plot lifecycle management across library and examples."""

from __future__ import annotations

import os
import sys
from contextlib import suppress
from typing import Any

from matplotlib.figure import Figure

from nirs4all.core.logging import get_logger

logger = get_logger(__name__)
_HEADLESS_WARNING_CONTEXTS: set[str] = set()

_NON_INTERACTIVE_BACKENDS = (
    "agg",
    "cairo",
    "pdf",
    "pgf",
    "ps",
    "svg",
    "template",
)


def normalize_figures(figures: Figure | list[Figure] | tuple[Figure, ...] | None) -> list[Figure]:
    """Normalize a figure or list of figures to a flat list."""
    if figures is None:
        return []
    if isinstance(figures, list | tuple):
        return [fig for fig in figures if fig is not None]
    return [figures]


def has_interactive_display() -> bool:
    """Return True when the current Matplotlib backend can show windows."""
    import matplotlib

    backend = str(matplotlib.get_backend()).lower()
    if any(token in backend for token in _NON_INTERACTIVE_BACKENDS):
        return False

    if sys.platform.startswith("linux") and not any(
        os.environ.get(name) for name in ("DISPLAY", "WAYLAND_DISPLAY", "MIR_SOCKET")
    ):
        return False

    return True


def close_figures(figures: Figure | list[Figure] | tuple[Figure, ...] | None) -> None:
    """Close one or more Matplotlib figures."""
    import matplotlib.pyplot as plt

    for fig in normalize_figures(figures):
        with suppress(Exception):
            plt.close(fig)


def finalize_figures(
    figures: Figure | list[Figure] | tuple[Figure, ...] | None,
    *,
    plots_visible: bool,
    figure_refs: list[Any] | None = None,
    close_when_hidden: bool = True,
) -> list[Figure]:
    """Keep figures alive for deferred display, or close them immediately."""
    figs = normalize_figures(figures)
    if not figs:
        return []

    if plots_visible:
        if figure_refs is not None:
            figure_refs.extend(figs)
        return figs

    if close_when_hidden:
        close_figures(figs)
    return figs


def show_figures(
    figures: Figure | list[Figure] | tuple[Figure, ...] | None,
    *,
    block: bool = True,
    close: bool = False,
    context: str = "plot",
) -> bool:
    """Show one or more figures once, in one place."""
    import matplotlib.pyplot as plt

    figs = normalize_figures(figures)
    if not figs:
        return False

    if not has_interactive_display() and context not in _HEADLESS_WARNING_CONTEXTS:
        logger.warning(
            f"Interactive {context} display requested, but the current Matplotlib backend "
            "or environment may be headless. Plots may not appear."
        )
        _HEADLESS_WARNING_CONTEXTS.add(context)

    try:
        plt.show(block=block)
        return True
    finally:
        if close:
            close_figures(figs)


def figure_list(figures: Figure | list[Figure] | tuple[Figure, ...] | None) -> list[Figure]:
    """Backward-compatible alias for normalize_figures()."""
    return normalize_figures(figures)


def keep_or_close_figures(
    figures: Figure | list[Figure] | tuple[Figure, ...] | None,
    *,
    visible: bool,
    figure_refs: list[Any] | None = None,
) -> list[Figure]:
    """Backward-compatible alias for finalize_figures()."""
    return finalize_figures(figures, plots_visible=visible, figure_refs=figure_refs)
