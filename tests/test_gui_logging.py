"""Terminal diagnostics for errors caught by the Streamlit UI."""

import logging
from io import StringIO

from django.test import SimpleTestCase
from rich.console import Console

from customer_service.gui_logging import (
    LOGGER_NAME,
    configure_gui_logger,
    make_rich_handler,
)
from customer_service.llm.client import LLMResponseError


class GUILoggingTests(SimpleTestCase):
    def test_repeated_configuration_installs_only_one_handler(self):
        logger = configure_gui_logger()
        self.assertIs(configure_gui_logger(), logger)
        self.assertEqual(
            sum(
                getattr(handler, "_data_analyst_handler", False)
                for handler in logger.handlers
            ),
            1,
        )
        self.assertEqual(logger.name, LOGGER_NAME)
        self.assertFalse(logger.propagate)

    def test_rich_traceback_has_error_id_but_not_frame_locals(self):
        output = StringIO()
        console = Console(
            file=output, force_terminal=False, color_system=None, width=120
        )
        logger = logging.getLogger("data_analyst.test_rich_output")
        handler = make_rich_handler(console)
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        logger.propagate = False
        try:

            def fail() -> None:
                secret_parts = ("DO", "NOT", "LOG", "THIS")
                secret = "_".join(secret_parts)
                raise LLMResponseError(
                    "LLM returned empty response content" if secret else "unreachable"
                )

            try:
                fail()
            except LLMResponseError:
                logger.exception("Analysis failed [error_id=abc12345]")
        finally:
            logger.removeHandler(handler)
            handler.close()

        rendered = output.getvalue()
        self.assertIn("error_id=abc12345", rendered)
        self.assertIn("LLMResponseError", rendered)
        self.assertIn("LLM returned empty response content", rendered)
        self.assertNotIn("DO_NOT_LOG_THIS", rendered)
