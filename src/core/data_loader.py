import logging
from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from ..utils import setup_logger

class DataLoader:
    def __init__(self, version="smote"):
        self.version = version
        self.logger = setup_logger('data_loader')
        
        # Protected attributes and groups
        self.protected_attributes = {
            'time': ['Time_Category_Morning', 'Time_Category_Night'],
            'location': ['Location_New York', 'Location_Philadelphia']
        }
        self.privileged_groups = [
            {'Time_Category_Morning': 1}, 
            {'Location_New York': 1}
        ]
        self.unprivileged_groups = [
            {'Time_Category_Night': 1},
            {'Location_Philadelphia': 1}
        ]

        # Initialize samplers
        self.samplers = {
            'smote': SMOTE(random_state=42),
            'ros': RandomOverSampler(random_state=42)
        }

    def _get_file_paths(self):
        # Validate and return file paths.
        base_path = Path('data/processed')
        required_files = {
            'X_train': base_path / f'X_train_{self.version}.csv',
            'X_test': base_path / 'X_test.csv',
            'y_train': base_path / f'y_train_{self.version}.csv',
            'y_test': base_path / 'y_test.csv'
        }
        
        # Check if files exist
        missing_files = [f for f, path in required_files.items() if not path.exists()]
        if missing_files:
            self.logger.error(f"Missing required files: {missing_files}")
            raise FileNotFoundError(f"Missing data files: {missing_files}")
            
        return required_files

    def _validate_data_shapes(self, X, y, dataset_name):
        # Validate that features and labels have matching lengths.
        if len(X) != len(y):
            self.logger.error(f"Shape mismatch in {dataset_name}: "
                            f"X has {len(X)} rows, y has {len(y)} rows")
            
            # Attempt automatic alignment by index if possible
            if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
                common_index = X.index.intersection(y.index)
                if len(common_index) > 0:
                    self.logger.warning(f"Aligning on {len(common_index)} common indices")
                    return X.loc[common_index], y.loc[common_index]
            
            raise ValueError(f"{dataset_name} shape mismatch: X({len(X)}) vs y({len(y)})")
        return X, y

    def _convert_labels(self, y):
        # Convert labels to numeric format with validation.
        if y.dtype == object:
            y = y.replace({'IsFraud': 1, 'NotFraud': 0})
            y = pd.to_numeric(y, errors='coerce')
            if y.isna().any():
                invalid_labels = y[y.isna()].index.tolist()
                self.logger.error(f"Invalid labels at indices: {invalid_labels[:10]}{'...' if len(invalid_labels) > 10 else ''}")
                raise ValueError("Label contains non-numeric values")
        return y.astype(int)

    def _check_class_distribution(self, y):
        # Validate and analyze class distribution.
        class_counts = np.bincount(y)
        self.logger.info(f"Class distribution: 0={class_counts[0]}, 1={class_counts[1]}")
        
        if len(class_counts) < 2:
            raise ValueError("Only one class present in training data")
        if class_counts[1] < 5:
            self.logger.warning(f"Very few fraud cases ({class_counts[1]}). Consider collecting more data.")
            
        return class_counts

    def _apply_sampling(self, X_train, y_train, class_counts):
        # Apply appropriate sampling technique based on version.
        if self.version == 'adasyn':
            n_neighbors = min(5, class_counts[1] - 1)
            if n_neighbors < 1:
                self.logger.warning("Not enough samples for ADASYN. Using SMOTE instead")
                return SMOTE(random_state=42).fit_resample(X_train, y_train)
                
            self.logger.info(f"Applying ADASYN with n_neighbors={n_neighbors}")
            return ADASYN(
                sampling_strategy='auto',
                n_neighbors=n_neighbors,
                random_state=42
            ).fit_resample(X_train, y_train)
            
        sampler = self.samplers.get(self.version)
        if sampler:
            return sampler.fit_resample(X_train, y_train)
            
        self.logger.warning(f"No sampler configured for version '{self.version}'. Returning original data.")
        return X_train, y_train

    def load_data(self):
        # Robust data loading with comprehensive validation.
        files = self._get_file_paths()
        
        # Load data with validation
        X_train = pd.read_csv(files['X_train'])
        y_train = pd.read_csv(files['y_train'], header=None).squeeze()
        X_test = pd.read_csv(files['X_test'])
        y_test = pd.read_csv(files['y_test'], header=None).squeeze()

        # Convert and validate labels
        y_train = self._convert_labels(y_train)
        y_test = self._convert_labels(y_test)

        # Validate shapes
        X_train, y_train = self._validate_data_shapes(X_train, y_train, 'train')
        X_test, y_test = self._validate_data_shapes(X_test, y_test, 'test')

        # Check class distribution
        class_counts = self._check_class_distribution(y_train)

        # Apply sampling
        X_train, y_train = self._apply_sampling(X_train, y_train, class_counts)

        self.logger.info(f"Final class distribution: {Counter(y_train)}")
        return (X_train, y_train), (X_test, y_test), self.privileged_groups, self.unprivileged_groups
