"""Application logging setup."""

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure concise console logging for the application."""

    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
    )
