"""Centralized logging configuration for GreenEdge-5G."""

import logging
import os
import sys


def setup_logging(
    level: str | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure and return the root logger for GreenEdge.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
               Defaults to GREENEDGE_LOG_LEVEL env var or INFO.
        format_string: Custom format string.

    Returns:
        Configured root logger.
    """
    if level is None:
        level = os.getenv("GREENEDGE_LOG_LEVEL", "INFO").upper()

    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"

    numeric_level = getattr(logging, level, logging.INFO)

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=format_string,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    return logging.getLogger("greenedge")


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: Module name (typically __name__).

    Returns:
        Logger instance for the module.
    """
    return logging.getLogger(f"greenedge.{name}")


# Initialize default logger on import
logger = setup_logging()
