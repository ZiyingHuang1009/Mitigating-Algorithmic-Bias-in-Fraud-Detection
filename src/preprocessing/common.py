import pandas as pd
from sklearn.preprocessing import StandardScaler
from ...config.constants import PROTECTED_ATTRS

def create_protected_features(df):
    # Generate dummy variables for protected attributes
    for attr in PROTECTED_ATTRS['time'] + PROTECTED_ATTRS['location']:
        if attr not in df.columns:
            category = attr.split('_')[1]
            col = 'TimeCategory' if 'Time' in attr else 'Location'
            df[attr] = (df[col] == category).astype(int)
    return df

def normalize_amounts(df):
    # Standard scaling for transaction amounts
    df['Amount'] = StandardScaler().fit_transform(df[['Amount']])
    return df