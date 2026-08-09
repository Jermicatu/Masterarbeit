# logger_config.py
import logging
import sys

def setup_logger(name="gaussian_product", log_file="debug.log"):
    """Create a logger that writes to both console and file."""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear any existing handlers (prevents duplicates)
    logger.handlers.clear()
    
    # File handler (overwrites on each run)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (optional, for real-time tracking)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Suppress torchquad logs
    logging.getLogger("torchquad").setLevel(logging.WARNING)
    logging.getLogger("torchquad").propagate = False
    
    return logger

# Export a global instance
logger = setup_logger()