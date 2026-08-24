"""Tests for user-facing logging behavior."""

import logging

from agentic_chatbot.logging_config import configure_logging


def test_info_logs_are_limited_to_the_application(caplog) -> None:
    root_logger = logging.getLogger()
    app_logger = logging.getLogger("agentic_chatbot.test")
    httpx_logger = logging.getLogger("httpx")
    httpcore_logger = logging.getLogger("httpcore")
    original_root_level = root_logger.level
    original_app_level = logging.getLogger("agentic_chatbot").level
    original_httpx_level = httpx_logger.level
    original_httpcore_level = httpcore_logger.level

    try:
        configure_logging("INFO")
        httpx_logger.info("HTTP Request: POST https://example.test")
        app_logger.info("Application message")

        assert "HTTP Request" not in caplog.text
        assert "Application message" in caplog.text
    finally:
        root_logger.setLevel(original_root_level)
        logging.getLogger("agentic_chatbot").setLevel(original_app_level)
        httpx_logger.setLevel(original_httpx_level)
        httpcore_logger.setLevel(original_httpcore_level)
