import logging
import os
import sys


def get_logger(name: str):
    level = os.environ.get("LOGLEVEL", "INFO").upper()
    logger = logging.getLogger(name)

    # Prevent double configuration by checking handlers
    if not logger.hasHandlers():
        logger.setLevel(level)
        # Stream handler to stdout
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter("[%(thread)d] %(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
