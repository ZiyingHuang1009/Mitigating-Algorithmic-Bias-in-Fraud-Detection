import os
import pandas as pd
from aif360.datasets import BinaryLabelDataset

def load_data():
    # Check files exist
    required_files = [
        'data/X_train.csv',
        'data/X_test.csv',
        'data/y_train.csv', 
        'data/y_test.csv'
    ]
    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Required file {file} not found")

    # Load data
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    # Combine into BinaryLabelDataset format
    train_data = pd.concat([X_train, pd.DataFrame(y_train, columns=['IsFraud'])], axis=1)
    test_data = pd.concat([X_test, pd.DataFrame(y_test, columns=['IsFraud'])], axis=1)

    # Define protected attribute (adjust based on your TimeCategory features)
    protected_attr = [col for col in X_train.columns if 'TimeCategory' in col][0]
    
    # Create dataset objects
    train_dataset = BinaryLabelDataset(
        df=train_data,
        label_names=['IsFraud'],
        protected_attribute_names=[protected_attr]
    )
    
    test_dataset = BinaryLabelDataset(
        df=test_data,
        label_names=['IsFraud'],
        protected_attribute_names=[protected_attr]
    )

    # Define privileged/unprivileged groups
    privileged = [{protected_attr: 0}]
    unprivileged = [{protected_attr: 1}]

    return train_dataset, test_dataset, privileged, unprivileged