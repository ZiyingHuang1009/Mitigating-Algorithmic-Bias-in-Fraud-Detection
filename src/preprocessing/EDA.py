import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from aif360.datasets import BinaryLabelDataset
import joblib
import os
import sys
from pathlib import Path
from datetime import datetime
import warnings
import logging

# Configure absolute imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.utils import setup_logger
from config.constants import PROTECTED_ATTRS

# Initialize logging
logger = setup_logger('eda_logger')

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="qt.qpa.fonts")

class FraudDataProcessor:
    def __init__(self):
        # Initialize the fraud data processor with proper validation
        self.raw_data = None
        self.preprocessor = None
        self.feature_names = []
        self.protected_attributes = self._validate_protected_attrs(PROTECTED_ATTRS)
        self._setup_directories()

    def _validate_protected_attrs(self, protected_attrs):
        # Validate protected attributes structure
        if not isinstance(protected_attrs, dict):
            logger.error("PROTECTED_ATTRS must be a dictionary")
            raise ValueError("PROTECTED_ATTRS must be a dictionary")
        
        required_keys = {'time', 'location'}
        missing_keys = required_keys - set(protected_attrs.keys())
        if missing_keys:
            logger.error(f"Missing required keys in PROTECTED_ATTRS: {missing_keys}")
            raise ValueError(f"Missing required keys in PROTECTED_ATTRS: {missing_keys}")
        
        return protected_attrs

    def _setup_directories(self):
        #  Ensure required directories exist with proper permissions
        required_dirs = {
            'data/processed': 0o755,
            'data/reports': 0o755,
            'models': 0o755
        }
        
        for dir_path, mode in required_dirs.items():
            try:
                os.makedirs(dir_path, mode=mode, exist_ok=True)
            except OSError as e:
                logger.error(f"Failed to create directory {dir_path}: {str(e)}")
                raise

    def load_data(self, data_path):
        # Load and validate raw data with comprehensive checks
        logger.info(f"Loading data from {data_path}")
        
        # Validate file existence
        if not os.path.isfile(data_path):
            logger.error(f"Data file not found at {data_path}")
            raise FileNotFoundError(f"Data file not found at {data_path}")
        
        # Define expected schema
        dtype_spec = {
            'TransactionID': 'str',
            'Amount': 'float32',
            'MerchantID': 'str',
            'TransactionType': 'category',
            'Location': 'category',
            'IsFraud': 'int8'
        }
        
        date_cols = ['TransactionDate']
        required_cols = set(dtype_spec.keys()).union(date_cols)
        
        # Load data with validation
        df = pd.read_csv(
            data_path,
            parse_dates=date_cols,
            dtype=dtype_spec,
            engine='c'
        )
        
        # Validate columns
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate data quality
        if df['IsFraud'].isna().any():
            logger.error("Found NA values in IsFraud column")
            raise ValueError("NA values found in IsFraud column")
        
        self.raw_data = df
        logger.info(f"Successfully loaded {len(df):,} records")
        return df

    def engineer_features(self, df):
        """Create temporal and protected features with proper initialization"""
        if not hasattr(self, 'raw_data') or self.raw_data is None:
            self.raw_data = df.copy()
            logger.info("Initialized raw_data reference")
        
        original_columns = set(df.columns)
        
        # Time features
        df['TransactionHour'] = df['TransactionDate'].dt.hour
        time_bins = [0, 6, 12, 18, 24]
        time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
        
        df['TimeCategory'] = pd.cut(
            df['TransactionHour'],
            bins=time_bins,
            labels=time_labels,
            ordered=False
        ).astype('category')
        
        # Protected attributes
        for time_cat in self.protected_attributes['time']:
            col_name = f'TimeCategory_{time_cat}'
            df[col_name] = (df['TimeCategory'] == time_cat).astype('int8')
            
        for loc in self.protected_attributes['location']:
            col_name = f'Location_{loc.replace(" ", "_")}'
            df[col_name] = (df['Location'] == loc).astype('int8')
            
        # Additional features
        df['LogAmount'] = np.log1p(df['Amount'])
        df['Weekend'] = (df['TransactionDate'].dt.dayofweek >= 5).astype('int8')
        
        new_features = set(df.columns) - original_columns
        logger.info(f"Added {len(new_features)} new features: {new_features}")
        return df

    def generate_visualizations(self, df):
        # Generate and save EDA visualizations with error handling
        logger.info("Generating EDA visualizations...")
        
        # Validate input
        if not isinstance(df, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            raise TypeError("Input must be a pandas DataFrame")
        
        required_cols = {'IsFraud', 'TimeCategory', 'Location', 'TransactionType'}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns for visualization: {missing_cols}")
            raise ValueError(f"Missing required columns for visualization: {missing_cols}")
        
        # Configure plot style
        plt.style.use('ggplot')
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({
            'font.size': 12,
            'figure.figsize': (14, 8),
            'axes.titlesize': 16,
            'axes.labelsize': 14
        })
        
        # Generate plots
        self._plot_class_distribution(df)
        self._plot_transaction_patterns(df)
        self._plot_protected_attributes(df)
        self._plot_correlation_matrix(df)
        
        logger.info("Visualizations saved to data/reports/")

    def _plot_class_distribution(self, df):
        # Plot distribution of fraud vs non-fraud cases
        plt.figure()
        fraud_pct = df['IsFraud'].mean() * 100
        ax = sns.countplot(x='IsFraud', data=df)
        plt.title(f'Class Distribution (Fraud: {fraud_pct:.2f}%)')
        plt.savefig('data/reports/class_distribution.png', bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_transaction_patterns(self, df):
        # Plot transaction patterns by time/location
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Hourly patterns
        hourly = df.groupby('TransactionHour')['IsFraud'].mean()
        sns.lineplot(x=hourly.index, y=hourly.values, ax=axes[0,0])
        axes[0,0].set(title='Fraud Rate by Hour', xlabel='Hour of Day', ylabel='Fraud Rate')
        
        # Time categories
        time_cat = df.groupby('TimeCategory')['IsFraud'].mean()
        sns.barplot(x=time_cat.index, y=time_cat.values, ax=axes[0,1])
        axes[0,1].set(title='Fraud by Time Category', xlabel='', ylabel='Fraud Rate')
        
        # Locations
        loc = df.groupby('Location')['IsFraud'].mean().sort_values()
        sns.barplot(x=loc.index, y=loc.values, ax=axes[1,0])
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].set(title='Fraud by Location', xlabel='', ylabel='Fraud Rate')
        
        # Transaction types
        txn_type = df.groupby('TransactionType')['IsFraud'].mean()
        sns.barplot(x=txn_type.index, y=txn_type.values, ax=axes[1,1])
        axes[1,1].set(title='Fraud by Transaction Type', xlabel='', ylabel='Fraud Rate')
        
        plt.tight_layout()
        plt.savefig('data/reports/transaction_patterns.png', bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_protected_attributes(self, df):
        # Analyze protected attributes distribution
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Time-based
        time_fraud = df.groupby('TimeCategory')['IsFraud'].mean()
        sns.barplot(x=time_fraud.index, y=time_fraud.values, ax=axes[0])
        axes[0].set(title='Fraud Rate by Time Category', xlabel='', ylabel='Fraud Rate')
        
        # Location-based
        loc_cols = [c for c in df.columns if c.startswith('Location_')]
        loc_fraud = df[loc_cols + ['IsFraud']].melt(id_vars='IsFraud').groupby('variable')['IsFraud'].mean()
        sns.barplot(x=loc_fraud.index, y=loc_fraud.values, ax=axes[1])
        axes[1].set(title='Fraud Rate by Location', xlabel='', ylabel='Fraud Rate')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('data/reports/fairness_analysis.png', bbox_inches='tight', dpi=300)
        plt.close()

    def _plot_correlation_matrix(self, df):
        # Plot correlation matrix of numerical features
        plt.figure(figsize=(12, 10))
        num_cols = df.select_dtypes(include=['number']).columns
        corr = df[num_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        plt.savefig('data/reports/correlation_matrix.png', bbox_inches='tight', dpi=300)
        plt.close()

    def preprocess_data(self, df):
        # Preprocess data for modeling with validation
        logger.info("Preprocessing data for modeling...")
        
        # Validate input
        if not isinstance(df, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            raise TypeError("Input must be a pandas DataFrame")
        
        if 'IsFraud' not in df.columns:
            logger.error("IsFraud column missing")
            raise ValueError("IsFraud column missing")
        
        # Feature selection
        features_to_drop = ['TransactionID', 'TransactionDate', 'TimeCategory', 'Location']
        X = df.drop(features_to_drop + ['IsFraud'], axis=1)
        y = df['IsFraud']
        
        # Define preprocessing
        numeric_features = ['Amount', 'TransactionHour', 'LogAmount', 'Weekend']
        categorical_features = ['TransactionType']
        
        # Create and fit preprocessor
        self.preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ], remainder='passthrough')
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        # Apply preprocessing
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        # Get feature names
        numeric_names = numeric_features
        categorical_names = list(self.preprocessor.named_transformers_['cat'].get_feature_names_out(['TransactionType']))
        remainder_names = [col for col in X.columns if col not in numeric_features + categorical_features]
        self.feature_names = numeric_names + categorical_names + remainder_names
        
        # Verify shape
        if X_train_processed.shape[1] != len(self.feature_names):
            logger.error(f"Column mismatch! Expected {len(self.feature_names)} features, got {X_train_processed.shape[1]}")
            raise ValueError(f"Column mismatch! Expected {len(self.feature_names)} features, got {X_train_processed.shape[1]}")
        
        # Handle class imbalance
        resampling_methods = {
            'original': (X_train_processed, y_train),
            'smote': SMOTE(random_state=42).fit_resample(X_train_processed, y_train),
            'adasyn': ADASYN(random_state=42).fit_resample(X_train_processed, y_train),
            'ros': RandomOverSampler(random_state=42).fit_resample(X_train_processed, y_train)
        }
        
        # Save processed data
        self._save_processed_data(resampling_methods, X_test_processed, y_test)
        
        return resampling_methods

    def _save_processed_data(self, resampling_results, X_test, y_test):
        # Save processed data to disk with validation
        logger.info("Saving processed data...")
        
        # Validate inputs
        if not isinstance(resampling_results, dict):
            logger.error("resampling_results must be a dictionary")
            raise TypeError("resampling_results must be a dictionary")
        
        if not isinstance(X_test, np.ndarray):
            logger.error("X_test must be a numpy array")
            raise TypeError("X_test must be a numpy array")
        
        # Save resampled versions
        for name, (X, y) in resampling_results.items():
            pd.DataFrame(X, columns=self.feature_names).to_csv(
                f'data/processed/X_train_{name}.csv', index=False)
            pd.DataFrame(y, columns=['IsFraud']).to_csv(
                f'data/processed/y_train_{name}.csv', index=False)
        
        # Save test data
        pd.DataFrame(X_test, columns=self.feature_names).to_csv(
            'data/processed/X_test.csv', index=False)
        pd.DataFrame(y_test, columns=['IsFraud']).to_csv(
            'data/processed/y_test.csv', index=False)
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'models/preprocessor.joblib')
        logger.info("Processed data saved successfully")

    def create_aif360_dataset(self, df):
        # Create properly formatted numerical dataset for AIF360 with validation
        logger.info("Creating AIF360 dataset...")
        
        # Validate input
        if not isinstance(df, pd.DataFrame):
            logger.error("Input must be a pandas DataFrame")
            raise TypeError("Input must be a pandas DataFrame")
        
        # Create a copy of relevant columns
        aif_df = df[['IsFraud']].copy().astype('float64')
        
        # Add protected attributes (ensure they're numerical)
        for time_cat in self.protected_attributes['time']:
            col_name = f'TimeCategory_{time_cat}'
            if col_name not in df.columns:
                logger.error(f"Protected attribute column {col_name} not found")
                raise ValueError(f"Protected attribute column {col_name} not found")
            aif_df[col_name] = df[col_name].astype('float64')
            
        for loc in self.protected_attributes['location']:
            col_name = f'Location_{loc.replace(" ", "_")}'
            if col_name not in df.columns:
                logger.error(f"Protected attribute column {col_name} not found")
                raise ValueError(f"Protected attribute column {col_name} not found")
            aif_df[col_name] = df[col_name].astype('float64')
        
        # Handle NA values
        if aif_df.isna().any().any():
            logger.warning("NA values detected - filling with 0")
            aif_df = aif_df.fillna(0)
        
        # Create dataset
        protected_attrs = [col for col in aif_df.columns 
                         if col.startswith('TimeCategory_') or col.startswith('Location_')]
        
        dataset = BinaryLabelDataset(
            df=aif_df,
            label_names=['IsFraud'],
            protected_attribute_names=protected_attrs,
            favorable_label=0,
            unfavorable_label=1
        )
        
        joblib.dump(dataset, 'models/aif360_dataset.joblib')
        logger.info("AIF360 dataset created and saved")
        return dataset

def run_pipeline(data_path='data/raw/credit_card_fraud_dataset.csv'):
    # Run the complete EDA and preprocessing pipeline with proper error handling
    processor = FraudDataProcessor()
    
    # Load data
    df = processor.load_data(data_path)
    
    # Feature engineering
    df = processor.engineer_features(df)
    
    # Generate visualizations
    processor.generate_visualizations(df)
    
    # Preprocess for modeling
    processor.preprocess_data(df)
    
    # Create AIF360 dataset
    processor.create_aif360_dataset(df)
    
    logger.info("Pipeline completed successfully")
    return True

if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)