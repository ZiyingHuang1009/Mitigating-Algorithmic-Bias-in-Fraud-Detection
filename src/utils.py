import logging
import os
import warnings
from datetime import datetime

def setup_logger(name=__name__, log_level=logging.INFO):
    
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times in case of repeated calls
    if not logger.handlers:
        logger.setLevel(log_level)
        
        # Create handlers
        os.makedirs('logs', exist_ok=True)
        log_file = os.path.join('logs', f'{datetime.now().strftime("%Y%m%d")}.log')
        
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add formatter to handlers
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

def create_directories():
    dirs = [
        'data/raw',
        'data/processed',
        'data/reports',
        'models',
        'results',
        'logs',
        'config'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def configure_environment():
    # Suppress TensorFlow info messages
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="inFairness")
    warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
    
    # For TensorFlow 2.x - simplified approach
    import importlib
    if importlib.util.find_spec("tensorflow"):
        import tensorflow as tf
        tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    # Initialize environment when run directly
    create_directories()
    configure_environment()
    logger = setup_logger()
    logger.info("Utility functions initialized")