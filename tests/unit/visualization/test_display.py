"""Unit tests for shared Matplotlib display helpers."""

import matplotlib.pyplot as plt

from nirs4all.visualization.display import finalize_figures, show_figures


class TestFinalizeFigures:
    """Test deferred figure retention behavior."""

    def test_tracks_visible_figures_for_deferred_display(self):
        """Visible figures should be retained instead of closed immediately."""
        fig = plt.figure()
        figure_refs = []

        try:
            result = finalize_figures(
                fig,
                plots_visible=True,
                figure_refs=figure_refs,
            )

            assert result == [fig]
            assert figure_refs == [fig]
            assert plt.fignum_exists(fig.number)
        finally:
            plt.close(fig)


class TestShowFigures:
    """Test centralized interactive display handling."""

    def test_skips_show_and_closes_when_backend_is_headless(self, monkeypatch):
        """Headless environments should not call plt.show()."""
        fig = plt.figure()
        closed = []
        original_close = plt.close

        def _fail_show(*args, **kwargs):
            raise AssertionError("show_figures should not call plt.show() in headless mode")

        def _record_close(target):
            closed.append(target)
            original_close(target)

        monkeypatch.setattr("nirs4all.visualization.display.has_interactive_display", lambda: False)
        monkeypatch.setattr(plt, "show", _fail_show)
        monkeypatch.setattr(plt, "close", _record_close)

        assert show_figures(fig, block=True, close=True) is False
        assert closed == [fig]
