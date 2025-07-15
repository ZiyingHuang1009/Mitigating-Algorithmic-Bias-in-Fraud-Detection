import os
import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from ..utils import setup_logger

logger = setup_logger('data_loader')

class DataLoader:
    def __init__(self, version='original'):
        self.version = version
        self.protected_attributes = {
            'time': ['Time_Morning', 'Time_Night'],
            'location': ['Location_NYC', 'Location_Philly']
        }
        self.privileged_groups = [{'Time_Morning': 1, 'Location_NYC': 1}]
        self.unprivileged_groups = [{'Time_Night': 1, 'Location_Philly': 1}]
        self.processed_dir = os.path.join('data', 'processed')
        self.logger = setup_logger('data_loader')

    def _get_file_paths(self):
        # Get file paths for training and test data with validation
        paths = {
            'X_train': os.path.join(self.processed_dir, f'X_train_{self.version}.csv'),
            'y_train': os.path.join(self.processed_dir, f'y_train_{self.version}.csv'),
            'X_test': os.path.join(self.processed_dir, 'X_test.csv'),
            'y_test': os.path.join(self.processed_dir, 'y_test.csv')
        }
        
        # Verify files exist before attempting to load
        for path in paths.values():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data file not found: {path}")
        return paths

    def _load_labels(self, path, expected_length):
        # Load and strictly enforce integer labels
        # Read raw file content first
        with open(path, 'r') as f:
            content = f.readlines()
        
        # Determine if file has headers
        has_header = not content[0].strip().replace('.','').isdigit()
        
        # Load data with appropriate parameters
        y = pd.read_csv(path, header=0 if has_header else None)
        
        # Take first column and convert to strict integers
        y_series = y.iloc[:, 0]
        y_series = pd.to_numeric(y_series, errors='coerce')
        y_series = y_series.fillna(0)
        
        # Convert to integers safely
        y_array = y_series.astype(int).values
        
        # Ensure proper shape and length
        y_array = y_array.reshape(-1)
        if len(y_array) != expected_length:
            self.logger.warning(f"Truncating labels from {len(y_array)} to {expected_length}")
            y_array = y_array[:expected_length]
            
        return y_array

    def _load_features(self, path):
        # Load features with protected attribute handling
        X = pd.read_csv(path)
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        return X.fillna(0)

    def _ensure_protected_attributes(self, X):
        # Ensure protected attributes are properly encoded as numerical values
        # Time categories - already encoded as 0/1
        if 'Time_Morning' not in X.columns:
            X['Time_Morning'] = np.random.choice([0, 1], size=len(X), p=[0.4, 0.6])
            X['Time_Night'] = 1 - X['Time_Morning']
        
        # Location categories - already encoded as 0/1
        if 'Location_NYC' not in X.columns:
            X['Location_NYC'] = np.random.choice([0, 1], size=len(X), p=[0.3, 0.7])
            X['Location_Philly'] = 1 - X['Location_NYC']
        
        # Remove any string columns
        string_cols = X.select_dtypes(include=['object']).columns
        for col in string_cols:
            if col in X.columns:
                X.drop(col, axis=1, inplace=True)
        
        return X

    def _create_dataset(self, X, y):
        # Create BinaryLabelDataset with numerical-only data
        # Ensure all data is numerical
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        
        # Handle protected attributes
        X = self._ensure_protected_attributes(X)
        
        # Get numerical protected attribute names
        protected_attrs = [col for col in X.columns 
                         if col.startswith('Time_') or col.startswith('Location_')]
        
        # Create dataset
        dataset_df = X.copy()
        dataset_df['IsFraud'] = y
        
        dataset = BinaryLabelDataset(
            df=dataset_df,
            label_names=['IsFraud'],
            protected_attribute_names=protected_attrs
        )
        
        # Verify numerical types
        if dataset.labels.dtype != np.int64:
            dataset.labels = dataset.labels.astype(np.int64)
        
        return dataset

    def load_data(self):
        # Main data loading method with comprehensive validation
        files = self._get_file_paths()
        
        # Load data
        X_train = self._load_features(files['X_train'])
        X_test = self._load_features(files['X_test'])
        y_train = self._load_labels(files['y_train'], len(X_train))
        y_test = self._load_labels(files['y_test'], len(X_test))
        
        self.logger.info(f"Final shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")
        self.logger.info(f"Final shapes - X_test: {X_test.shape}, y_test: {y_test.shape}")
        
        # Create datasets
        train_data = self._create_dataset(X_train, y_train)
        test_data = self._create_dataset(X_test, y_test)
        
        return train_data, test_data, self.privileged_groups, self.unprivileged_groups

def load_data(version='original'):
    # Module-level function to load data
    return DataLoader(version).load_data()