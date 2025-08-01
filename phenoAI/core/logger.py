"""
Logging configuration for PhenoAI package.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = 'phenoai',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: int = 10*1024*1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logger with console and optional file output.
    
    Args:
        name: Logger name
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional log file path
        log_format: Custom log format string
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Default format
    if log_format is None:
        log_format = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(filename)s:%(lineno)d - %(message)s'
        )
    
    formatter = logging.Formatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, level.upper()))
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str = 'phenoai') -> logging.Logger:
    """Get existing logger or create default one."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger

class LoggerMixin:
    """Mixin class to add logging capability to other classes."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f'phenoai.{self.__class__.__name__}')
        return self._logger
