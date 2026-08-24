"""Minimal executable entry point for the project foundation."""

import logging

from agentic_chatbot.config import load_settings
from agentic_chatbot.logging_config import configure_logging

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Load configuration and confirm that the application can start."""

    settings = load_settings()
    configure_logging(settings.log_level)
    LOGGER.info("Agentic AI chatbot foundation started successfully.")


if __name__ == "__main__":
    main()
