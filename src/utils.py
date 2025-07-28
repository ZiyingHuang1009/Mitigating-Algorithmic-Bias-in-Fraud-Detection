import logging
import os
import warnings
from datetime import datetime

def setup_logger(name=__name__, log_level=logging.INFO):
    # Configure and return a logger.
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(log_level)
        os.makedirs('logs', exist_ok=True)
        
        # File handler with daily logs
        file_handler = logging.FileHandler(
            os.path.join('logs', f'{datetime.now().strftime("%Y%m%d")}.log')
        )
        # Console handler
        console_handler = logging.StreamHandler()
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def configure_environment():
    # Configure runtime environment.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
