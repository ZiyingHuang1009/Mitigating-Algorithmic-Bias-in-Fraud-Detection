import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.time_features = [
            'hour_of_day',
            'day_of_week',
            'time_since_last_txn'
        ]
    
    def transform(self, X, y=None):
        # Add temporal and behavioral features
        X = X.copy()
        
        # Temporal features
        if 'Timestamp' in X.columns:
            X['hour_of_day'] = X['Timestamp'].dt.hour
            X['day_of_week'] = X['Timestamp'].dt.dayofweek
            X['time_since_last_txn'] = X.groupby('UserID')['Timestamp'].diff().dt.total_seconds()
        
        # Behavioral features
        if 'Amount' in X.columns:
            X['amount_deviation'] = X['Amount'] - X.groupby('UserID')['Amount'].transform('mean')
            X['rolling_3h_spend'] = X.groupby('UserID')['Amount'].rolling('3h').sum().values
        
        return X, y if y is not None else X