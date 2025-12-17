"""Tests for nirs4all logging module."""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from nirs4all.core.logging import (
    BranchContext,
    ConsoleFormatter,
    FileFormatter,
    JsonFormatter,
    LogContext,
    Phase,
    Status,
    Symbols,
    ThrottledHandler,
    configure_logging,
    format_duration,
    format_number,
    format_run_footer,
    format_run_header,
    format_table,
    get_config,
    get_logger,
    get_run_id,
    is_configured,
    reset_logging,
)


class TestSymbols:
    """Test symbol system."""

    def test_unicode_symbols(self) -> None:
        """Test Unicode symbols."""
        symbols = Symbols(use_unicode=True)
        assert symbols.starting == ">"
        assert symbols.success == "[OK]"
        assert symbols.warning == "[!]"
        assert symbols.error == "[X]"

    def test_ascii_symbols(self) -> None:
        """Test ASCII symbols (same as Unicode per spec)."""
        symbols = Symbols(use_unicode=False)
        assert symbols.starting == ">"
        assert symbols.success == "[OK]"
        assert symbols.warning == "[!]"
        assert symbols.error == "[X]"

    def test_status_symbol_mapping(self) -> None:
        """Test status to symbol mapping."""
        symbols = Symbols()
        assert symbols.get_status_symbol(Status.SUCCESS) == "[OK]"
        assert symbols.get_status_symbol(Status.WARNING) == "[!]"
        assert symbols.get_status_symbol(Status.ERROR) == "[X]"
        assert symbols.get_status_symbol(Status.STARTING) == ">"
        assert symbols.get_status_symbol(None) == ""


class TestFormatters:
    """Test log formatters."""

    def test_format_duration_seconds(self) -> None:
        """Test duration formatting for seconds."""
        assert format_duration(0.5) == "0.5s"
        assert format_duration(5.0) == "5.0s"
        assert format_duration(59.9) == "59.9s"

    def test_format_duration_minutes(self) -> None:
        """Test duration formatting for minutes."""
        assert format_duration(60) == "1m 0.0s"
        assert format_duration(125.4) == "2m 5.4s"
        assert format_duration(3599) == "59m 59.0s"

    def test_format_duration_hours(self) -> None:
        """Test duration formatting for hours."""
        assert format_duration(3600) == "1h 0m 0s"
        assert format_duration(7325) == "2h 2m 5s"

    def test_format_number_int(self) -> None:
        """Test number formatting for integers."""
        assert format_number(1000) == "1,000"
        assert format_number(1234567) == "1,234,567"

    def test_format_number_float(self) -> None:
        """Test number formatting for floats."""
        assert format_number(0.381) == "0.381"
        assert format_number(0.12345678) == "0.123"

    def test_format_table(self) -> None:
        """Test table formatting."""
        headers = ["Name", "Score"]
        rows = [["snv", "0.405"], ["msc", "0.392"]]
        table = format_table(headers, rows)

        assert "+------+-------+" in table
        assert "| Name " in table
        assert "| snv " in table
        assert "| msc " in table

    def test_format_run_header(self) -> None:
        """Test run header formatting."""
        header = format_run_header(
            run_name="test_run",
            start_time=datetime(2025, 12, 16, 19, 12, 3),
            environment_info={"Python": "3.13"},
            reproducibility_info={"seed": "42"},
            use_unicode=False,
        )

        assert "nirs4all run: test_run" in header
        assert "2025-12-16 19:12:03" in header
        assert "Python: 3.13" in header
        assert "seed=42" in header
        assert "=" * 80 in header

    def test_format_run_footer(self) -> None:
        """Test run footer formatting."""
        footer = format_run_footer(
            status=Status.SUCCESS,
            duration_seconds=125.4,
            best_pipeline="SavGol -> PLS",
            metrics={"RMSE": 0.381, "R2": 0.85},
            use_unicode=False,
        )

        assert "[OK]" in footer
        assert "2m 5.4s" in footer
        assert "SavGol -> PLS" in footer
        assert "RMSE=" in footer
        assert "=" * 80 in footer


class TestConsoleFormatter:
    """Test console formatter."""

    def test_basic_format(self) -> None:
        """Test basic message formatting."""
        formatter = ConsoleFormatter(use_colors=False, use_unicode=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        assert "Test message" in result

    def test_status_format(self) -> None:
        """Test message with status."""
        formatter = ConsoleFormatter(use_colors=False, use_unicode=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Done",
            args=(),
            exc_info=None,
        )
        record.status = Status.SUCCESS

        result = formatter.format(record)
        assert "[OK]" in result
        assert "Done" in result


class TestJsonFormatter:
    """Test JSON formatter."""

    def test_basic_json(self) -> None:
        """Test basic JSON output."""
        import json

        formatter = JsonFormatter(run_id="test-123")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["run_id"] == "test-123"
        assert "ts" in data


class TestLogContext:
    """Test log context management."""

    def test_run_context(self) -> None:
        """Test run context creation."""
        with LogContext(run_id="test-run", run_name="Test Run", project="test"):
            run_id = get_run_id()
            assert run_id == "test-run"

        # Outside context
        assert get_run_id() is None

    def test_branch_context(self) -> None:
        """Test branch context."""
        with LogContext(run_id="test-run"):
            with LogContext.branch("snv", index=0, total=4) as branch:
                assert branch.name == "snv"
                assert branch.index == 0
                assert branch.total == 4
                assert branch.depth == 0

    def test_nested_branch_context(self) -> None:
        """Test nested branch context."""
        with LogContext(run_id="test-run"):
            with LogContext.branch("outer", index=0, total=2):
                with LogContext.branch("inner", index=0, total=3) as inner:
                    assert inner.name == "inner"
                    assert inner.depth == 1
                    assert inner.path == ["outer", "inner"]

    def test_source_context(self) -> None:
        """Test source context."""
        with LogContext(run_id="test-run"):
            with LogContext.source("NIR", index=0, total=3) as source:
                assert source.name == "NIR"
                assert source.index == 0
                assert source.total == 3

    def test_stack_context(self) -> None:
        """Test stacking context."""
        with LogContext(run_id="test-run"):
            with LogContext.stack(
                n_branches=4,
                meta_model="Ridge",
                branch_sources=["snv", "msc"],
            ) as stack:
                assert stack.n_branches == 4
                assert stack.meta_model == "Ridge"
                assert stack.branch_sources == ["snv", "msc"]


class TestConfiguration:
    """Test logging configuration."""

    def setup_method(self) -> None:
        """Reset logging before each test."""
        reset_logging()

    def teardown_method(self) -> None:
        """Reset logging after each test."""
        reset_logging()

    def test_configure_logging(self) -> None:
        """Test basic configuration."""
        configure_logging(verbose=1, use_unicode=False, use_colors=False)

        assert is_configured()
        config = get_config()
        assert config.verbose == 1
        assert config.use_unicode is False
        assert config.use_colors is False

    def test_get_logger(self) -> None:
        """Test logger retrieval."""
        configure_logging(verbose=2)
        logger = get_logger("test_module")

        assert logger is not None
        assert logger.name.startswith("nirs4all")

    def test_logger_methods(self) -> None:
        """Test logger convenience methods exist."""
        configure_logging(verbose=2)
        logger = get_logger("test_module")

        # Check methods exist
        assert hasattr(logger, "success")
        assert hasattr(logger, "starting")
        assert hasattr(logger, "progress")
        assert hasattr(logger, "metric")
        assert hasattr(logger, "artifact")
        assert hasattr(logger, "phase_start")
        assert hasattr(logger, "phase_complete")


class TestThrottledHandler:
    """Test throttled handler."""

    def test_milestone_passthrough(self) -> None:
        """Test that milestone percentages always pass through."""
        records: list[logging.LogRecord] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        base = CaptureHandler()
        handler = ThrottledHandler(base, min_interval=10.0)

        # Create progress records at milestone percentages
        for pct in [10, 25, 50, 75, 90, 100]:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"Progress {pct}%",
                args=(),
                exc_info=None,
            )
            record.is_progress = True
            record.percentage = pct
            record.is_new_best = False
            handler.emit(record)

        # All milestones should pass through
        assert len(records) == 6

    def test_new_best_passthrough(self) -> None:
        """Test that new best results always pass through."""
        records: list[logging.LogRecord] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        base = CaptureHandler()
        handler = ThrottledHandler(base, min_interval=10.0)

        # Create two new best records
        for i in range(2):
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=f"New best {i}",
                args=(),
                exc_info=None,
            )
            record.is_progress = True
            record.percentage = 15  # Not a milestone
            record.is_new_best = True
            handler.emit(record)

        assert len(records) == 2


class TestFileLogging:
    """Test file logging functionality."""

    def test_file_handler_creates_file(self) -> None:
        """Test that file handler creates log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            reset_logging()
            configure_logging(
                verbose=2,
                log_file=True,
                log_dir=tmpdir,
                run_id="test-file-log",
            )

            logger = get_logger("test_file")
            logger.info("Test file logging")

            # Check file was created
            log_file = Path(tmpdir) / "test-file-log.log"
            assert log_file.exists()

            reset_logging()
