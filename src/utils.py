import logging
import os
import warnings
from datetime import datetime
import sys

def setup_logger(name):
    # Configure and return a logger instance.
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    
    # Avoid adding multiple handlers
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger

def configure_environment():
    # Configure runtime environment.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
