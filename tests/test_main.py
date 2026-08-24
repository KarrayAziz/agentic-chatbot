"""Tests for the minimal executable entry point."""

import logging

from agentic_chatbot.main import main


def test_main_starts_successfully(caplog) -> None:
    with caplog.at_level(logging.INFO):
        main()

    assert "Agentic AI chatbot foundation started successfully." in caplog.text
