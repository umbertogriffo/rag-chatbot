import logging
import sys

from core.config import settings


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    # Prevent double configuration by checking handlers
    if not logger.hasHandlers():
        logger.setLevel(settings.LOG_LEVEL)
        # Stream handler to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(settings.LOG_LEVEL)
        formatter = logging.Formatter("[%(thread)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def experimental(func):
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__name__)
        logger.warning(f"{func.__name__} is an experimental function and may change in the future.")
        return func(*args, **kwargs)

    return wrapper
