"""Application logging setup."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure app logging without exposing noisy dependency request logs."""

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("agentic_chatbot").setLevel(level)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
