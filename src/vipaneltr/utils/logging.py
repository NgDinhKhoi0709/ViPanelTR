"""
Logging utilities for PanelTR-ViTabQA.

Provides structured logging with file and console output.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Global logger cache
_loggers: dict = {}


def setup_logger(
    level: str = "INFO",
    output_dir: Optional[str] = None,
    log_format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
) -> logging.Logger:
    """
    Setup the root logger for PanelTR.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        output_dir: Directory to save log files (optional)
        log_format: Log message format
        
    Returns:
        Root logger instance
    """
    # Get root logger for the package
    logger = logging.getLogger("vipaneltr")
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Set level
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # Create formatter
    formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler disabled; keep console-only logging
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Logger instance
    """
    if name in _loggers:
        return _loggers[name]
    
    # Create child logger under vipaneltr.
    if not name.startswith("vipaneltr"):
        name = f"vipaneltr.{name}"
    
    logger = logging.getLogger(name)
    _loggers[name] = logger
    
    return logger


class LogContext:
    """Context manager for adding context to log messages."""
    
    def __init__(self, logger: logging.Logger, context: str):
        """
        Initialize log context.
        
        Args:
            logger: Logger instance
            context: Context string to prepend
        """
        self.logger = logger
        self.context = context
    
    def debug(self, msg: str, *args, **kwargs):
        self.logger.debug(f"[{self.context}] {msg}", *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self.logger.info(f"[{self.context}] {msg}", *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self.logger.warning(f"[{self.context}] {msg}", *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self.logger.error(f"[{self.context}] {msg}", *args, **kwargs)
