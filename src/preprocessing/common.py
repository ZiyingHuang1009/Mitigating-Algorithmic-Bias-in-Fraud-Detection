import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TemporalBinner(BaseEstimator, TransformerMixin):
    # Handles time binning from TransactionDate
    def __init__(self, time_col='TransactionDate'):
        self.time_col = time_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.time_col in X.columns:
            # Handle both 'HH:MM.S' and 'YYYY-MM-DD HH:MM' formats
            if X[self.time_col].str.contains('-').any():  # Contains date portion
                hours = pd.to_datetime(X[self.time_col]).dt.hour
            else:  # Just time portion
                hours = X[self.time_col].str.split(':').str[0].astype(int)
            X['Time_Bin'] = pd.cut(hours, bins=[0,6,12,18,24], 
                                  labels=['Night','Morning','Afternoon','Evening'],
                                  right=False)
        return X

class AmountScaler(BaseEstimator, TransformerMixin):
    # Normalizes transaction amounts
    def __init__(self, method='minmax'):
        self.method = method
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if 'Amount' in X.columns:
            if self.method == 'minmax':
                X['Amount'] = (X['Amount'] - X['Amount'].min()) / (X['Amount'].max() - X['Amount'].min())
            elif self.method == 'log':
                X['Amount'] = np.log1p(X['Amount'])
        return X
