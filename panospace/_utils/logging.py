"""
PanoSpace Logging Utility
=========================
Provides a standard logging setup for consistent logging across PanoSpace modules.
"""

import logging


def setup_logger(
    name: str = "panospace",
    level: int = logging.INFO,
    fmt: str = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """Configure and return a logger with a standardized format.

    Parameters
    ----------
    name : str, optional
        Name of the logger, by default "panospace"
    level : int, optional
        Logging level, by default logging.INFO
    fmt : str, optional
        Format string for log messages, by default includes timestamp, level, and message
    datefmt : str, optional
        Date format for timestamps, by default "%Y-%m-%d %H:%M:%S"

    Returns
    -------
    logging.Logger
        Configured logger object.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt, datefmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
