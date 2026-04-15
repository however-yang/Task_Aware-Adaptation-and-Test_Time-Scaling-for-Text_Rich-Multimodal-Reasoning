from __future__ import annotations
import logging
import sys

def get_logger(name: str = "text_rich_mllm", level: int = logging.INFO) -> logging.Logger:
    """Setup and retrieve a configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    return logger
