"""Terminal diagnostics for Streamlit analysis failures."""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

LOGGER_NAME = "sales_data_analyst.gui"
_HANDLER_MARKER = "_sales_data_analyst_handler"


def make_rich_handler(console: Console | None = None) -> RichHandler:
    """Build a traceback handler without exposing frame-local data."""

    handler = RichHandler(
        console=console or Console(stderr=True),
        rich_tracebacks=True,
        tracebacks_show_locals=False,
        show_path=False,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    setattr(handler, _HANDLER_MARKER, True)
    return handler


def configure_gui_logger() -> logging.Logger:
    """Install one terminal handler despite Streamlit's script reruns."""

    logger = logging.getLogger(LOGGER_NAME)
    if not any(getattr(handler, _HANDLER_MARKER, False) for handler in logger.handlers):
        logger.addHandler(make_rich_handler())
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    return logger


def log_analysis_failure(exc: Exception, *, error_id: str) -> None:
    """Print an identifiable Rich traceback without logging request payloads."""

    configure_gui_logger().error(
        "Analysis failed [error_id=%s]",
        error_id,
        exc_info=(type(exc), exc, exc.__traceback__),
    )
