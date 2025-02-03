import sys
from loguru import logger

def configure_logging(log_level="INFO", log_json=False):
    """Configure logging in a way that works with both Loguru and Pyomo.
    
    Args:
        log_level: The logging level to use
        log_json: Whether to serialize logging as JSON
    """
    # Remove all existing handlers
    logger.remove()
    
    # Add a single handler to stdout with appropriate configuration
    logger.add(
        sys.__stdout__,
        level=log_level,
        serialize=log_json,
        backtrace=True,
        diagnose=True,
        enqueue=True,  # Use queue for thread-safe logging
        catch=True,    # Catch exceptions in the logging handler
    )
    
    return logger

class ThreadSafeStreamToLogger:
    """Thread-safe version of StreamToLogger."""
    
    def __init__(self, level="INFO"):
        self._level = level
        
    def write(self, buffer):
        if buffer.strip():  # Only log non-empty lines
            logger.opt(depth=1).log(self._level, buffer.rstrip())
    
    def flush(self):
        pass 