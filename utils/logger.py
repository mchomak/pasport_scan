"""Logging configuration."""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any
import structlog

# Directory for debug log files
_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")


def setup_logger(log_level: str = "INFO") -> None:
    """Configure structured logging with structlog + file debug handler."""

    os.makedirs(_LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler — respects configured log_level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, log_level.upper()))
    console.setFormatter(logging.Formatter("%(message)s"))

    # File handler — always DEBUG, rotates at 10 MB, keeps 5 files
    file_handler = RotatingFileHandler(
        os.path.join(_LOG_DIR, "debug.log"),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )

    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Use structlog's ProcessorFormatter for console output
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.dev.ConsoleRenderer(colors=False),
    )
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)


def get_logger(name: str) -> Any:
    """Get a logger instance."""
    return structlog.get_logger(name)


def get_file_logger(name: str) -> logging.Logger:
    """Get a stdlib logger that writes to the debug log file."""
    return logging.getLogger(name)
