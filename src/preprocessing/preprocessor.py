import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.pipeline import make_pipeline as make_imb_pipeline

class TemporalBinner(BaseEstimator, TransformerMixin):
    # Extracts temporal features from transaction timestamp
    def __init__(self, time_col='TransactionDate'):
        self.time_col = time_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.time_col not in X.columns:
            return X
            
        if pd.api.types.is_datetime64_any_dtype(X[self.time_col]):
            dt = X[self.time_col]
        else:
            dt = pd.to_datetime(X[self.time_col], errors='coerce')
            if dt.isna().any():
                raise ValueError(f"Invalid datetime values in {self.time_col}")
        
        X['Hour'] = dt.dt.hour
        X['Weekend'] = (dt.dt.dayofweek >= 5).astype(int)
        X['Time_Category'] = pd.cut(
            dt.dt.hour,
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            right=False
        )
        return X.drop(columns=[self.time_col])

class AmountTransformer(BaseEstimator, TransformerMixin):
    # Applies log transform and scaling to transaction amounts
    def __init__(self):
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        if 'Amount' in X.columns:
            valid_amounts = pd.to_numeric(X['Amount'], errors='coerce').dropna()
            if len(valid_amounts) < len(X):
                raise ValueError("Invalid numeric values in Amount column")
            amounts = np.log1p(valid_amounts).values.reshape(-1, 1)
            self.scaler.fit(amounts)
        return self
        
    def transform(self, X):
        X = X.copy()
        if 'Amount' in X.columns:
            X['LogAmount'] = self.scaler.transform(
                np.log1p(X['Amount']).values.reshape(-1, 1)
            )
        return X.drop(columns=['Amount'], errors='ignore')

class FraudDataPreprocessor:
    # Complete preprocessing pipeline for fraud detection
    def __init__(self, resampling_method='SMOTE', test_size=0.2, random_state=42):
        if resampling_method not in ['SMOTE', 'ADASYN', 'ROS']:
            raise ValueError("resampling_method must be one of: SMOTE, ADASYN, ROS")
        
        self.resampling_method = resampling_method
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names_ = None
        
    def _get_resampler(self):
        return {
            'SMOTE': SMOTE(random_state=self.random_state),
            'ADASYN': ADASYN(random_state=self.random_state),
            'ROS': RandomOverSampler(random_state=self.random_state)
        }[self.resampling_method]
    
    def _validate_data(self, df):
        required_cols = ['TransactionDate', 'Amount', 'Location', 'IsFraud', 'TransactionType']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert IsFraud to boolean
        df['IsFraud'] = df['IsFraud'].astype(int)
        
        # Validate Amount
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        if df['Amount'].isna().any():
            raise ValueError("Amount contains invalid numeric values")
            
        return df.dropna(subset=['Amount', 'IsFraud'])

    def _build_preprocessor(self):
        numeric_features = ['Hour', 'Weekend', 'LogAmount']
        categorical_features = ['Location', 'TransactionType', 'Time_Category']
        
        preprocessor = ColumnTransformer([
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
        
        return make_imb_pipeline(
            AmountTransformer(),
            TemporalBinner(),
            preprocessor
        )

    def preprocess(self, df):
        df_clean = self._validate_data(df.copy())
        
        X = df_clean.drop('IsFraud', axis=1)
        y = df_clean['IsFraud']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Create and fit preprocessor
        preprocessor = self._build_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Get feature names
        num_features = ['Hour', 'Weekend', 'LogAmount']
        cat_encoder = preprocessor.named_steps['columntransformer'].named_transformers_['cat']
        cat_features = cat_encoder.get_feature_names_out()
        self.feature_names_ = num_features + list(cat_features)
        
        # Apply resampling
        resampler = self._get_resampler()
        X_train_res, y_train_res = resampler.fit_resample(
            pd.DataFrame(X_train_processed, columns=self.feature_names_),
            y_train
        )
        
        # Transform test data
        X_test_processed = pd.DataFrame(
            preprocessor.transform(X_test),
            columns=self.feature_names_
        )
        
        return X_train_res, y_train_res, X_test_processed, y_test

def save_processed_data(X_train=None, y_train=None, X_test=None, y_test=None, method=None):
    # Save processed data with proper formatting
    output_dir = Path('data/processed/')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train data
    if X_train is not None and method is not None:
        X_train.to_csv(output_dir / f'X_train_{method.lower()}.csv', 
                      index=False, 
                      float_format='%.6f')
        y_train.to_csv(output_dir / f'y_train_{method.lower()}.csv', 
                      index=False)
    
    # Save test data (only once)
    if X_test is not None and not (output_dir / 'X_test.csv').exists():
        X_test.to_csv(output_dir / 'X_test.csv', 
                     index=False, 
                     float_format='%.6f')
        y_test.to_csv(output_dir / 'y_test.csv', 
                     index=False)

def main():
    # Configuration
    raw_data_path = Path('data/raw/credit_card_fraud_dataset.csv')
    processed_dir = Path('data/processed/')
    
    # Validate input
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Input file not found at {raw_data_path}")
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(raw_data_path)
    
    # Create shared test set first
    print("Creating test set...")
    test_processor = FraudDataPreprocessor()
    _, _, X_test, y_test = test_processor.preprocess(df)
    save_processed_data(X_test=X_test, y_test=y_test)
    
    # Process each resampling method
    for method in ['SMOTE', 'ADASYN', 'ROS']:
        print(f"Processing with {method}...")
        processor = FraudDataPreprocessor(resampling_method=method)
        X_train, y_train, _, _ = processor.preprocess(df)
        save_processed_data(X_train=X_train, y_train=y_train, method=method)
    
    print("\nProcessing completed successfully!")
    print("Generated files:")
    print(f"- {processed_dir/'X_test.csv'}")
    print(f"- {processed_dir/'y_test.csv'}")
    for method in ['smote', 'adasyn', 'ros']:
        print(f"- {processed_dir/f'X_train_{method}.csv'}")
        print(f"- {processed_dir/f'y_train_{method}.csv'}")

if __name__ == '__main__':
    main()
