from aif360.datasets import BinaryLabelDataset
import joblib
from ..utils import setup_logger

logger = setup_logger('preprocessor')

def load_aif360_dataset():
    # Load preprocessed AIF360 dataset
    try:
        dataset = joblib.load('models/aif360_dataset.joblib')
        protected = {
            'time': [attr.split('_')[1] for attr in dataset.protected_attribute_names if 'TimeCategory' in attr],
            'location': [attr.split('_')[1] for attr in dataset.protected_attribute_names if 'Location' in attr]
        }
        return dataset, protected
    except Exception as e:
        logger.error(f"Failed to load AIF360 dataset: {str(e)}")
        raise

def validate_dataset(dataset):
    # Validate dataset structure
    required = ['instance_weights', 'labels', 'features']
    if not all(hasattr(dataset, attr) for attr in required):
        raise ValueError("Invalid dataset structure")
    return True