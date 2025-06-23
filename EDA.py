import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib

# Configuration
sns.set_style("whitegrid")
plt.rcParams.update({
    'font.family': 'Arial',
    'figure.figsize': (12, 8)
})

def load_and_preprocess():
    df = pd.read_csv('credit_card_fraud_dataset.csv')
    
    print("\nInitial missing values:")
    print(df.isnull().sum())
    
    # Convert TransactionDate first
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
    
    # Create time features (handle NaT from conversion errors)
    df['TransactionHour'] = df['TransactionDate'].dt.hour
    df['TransactionHour'] = df['TransactionHour'].fillna(df['TransactionHour'].median())
    
    # Handle numeric columns
    numeric_cols = ['Amount', 'MerchantID']
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            print(f"\nFilling missing values in {col}:")
            print(f"Before: {df[col].isnull().sum()} missing")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
            print(f"After: {df[col].isnull().sum()} missing")
    
    # Handle categorical columns
    categorical_cols = ['TransactionType', 'Location']
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            print(f"\nFilling missing values in {col}:")
            print(f"Before: {df[col].isnull().sum()} missing")
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 'Unknown')
            print(f"After: {df[col].isnull().sum()} missing")
    
    # Create time categories
    time_bins = [0, 6, 12, 18, 24]
    time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
    df['TimeCategory'] = pd.cut(df['TransactionHour'], bins=time_bins, labels=time_labels)
    df['TimeCategory'] = df['TimeCategory'].fillna('Afternoon')  # Default fill
    
    # Final verification
    print("\nFinal missing values check:")
    print(df.isnull().sum())
    
    return df

def generate_eda_plots(df):
    # 1. Main Distribution Plots
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 12))
    
    # Transaction Amount Distribution
    sns.histplot(df['Amount'], kde=True, bins=30, ax=axes1[0, 0])
    axes1[0, 0].set_title('Transaction Amount Distribution')
    
    # Class Distribution
    fraud_counts = df['IsFraud'].value_counts()
    axes1[0, 1].pie(fraud_counts, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%')
    axes1[0, 1].set_title('Class Distribution')
    
    # Transaction Type Analysis
    sns.countplot(x='TransactionType', hue='IsFraud', data=df, ax=axes1[1, 0])
    axes1[1, 0].set_title('Transaction Type by Fraud Status')
    
    # Location Analysis
    top_locs = df['Location'].value_counts().nlargest(5).index
    sns.countplot(x='Location', hue='IsFraud', data=df[df['Location'].isin(top_locs)], 
                 order=top_locs, ax=axes1[1, 1])
    axes1[1, 1].set_title('Top 5 Locations by Fraud Status')
    axes1[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('eda_distributions.png')
    plt.close()

    # 2. Time Analysis
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    # Hourly Fraud Rate
    hourly_rate = df.groupby('TransactionHour')['IsFraud'].mean()
    sns.lineplot(x=hourly_rate.index, y=hourly_rate.values, ax=axes2[0])
    axes2[0].set_title('Hourly Fraud Rate')
    
    # Time Category Fraud Rate
    time_rate = df.groupby('TimeCategory')['IsFraud'].mean().reset_index()
    sns.barplot(x='TimeCategory', y='IsFraud', data=time_rate, ax=axes2[1])
    axes2[1].set_title('Fraud Rate by Time Category')
    
    plt.tight_layout()
    plt.savefig('time_analysis.png')
    plt.close()

    return df

def preprocess_data(df):
    # Feature selection
    X = df.drop(['TransactionID', 'TransactionDate', 'IsFraud'], axis=1)
    y = df['IsFraud']
    
    # Ensure proper data types
    numeric_cols = ['Amount', 'MerchantID', 'TransactionHour']
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Define transformers
    numeric_features = ['Amount', 'MerchantID', 'TransactionHour']
    categorical_features = ['TransactionType', 'Location', 'TimeCategory']
    
    # Only use existing columns
    numeric_features = [col for col in numeric_features if col in X.columns]
    categorical_features = [col for col in categorical_features if col in X.columns]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    # Transform the data
    X_train_preprocessed = preprocessor.transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Handle imbalance
    print("\nClass distribution before SMOTE:")
    print(y_train.value_counts())
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_preprocessed, y_train)
    
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_smote).value_counts())

    # Save processed data
    joblib.dump({
        'X_train': X_train_smote,
        'X_test': X_test_preprocessed,
        'y_train': y_train_smote,
        'y_test': y_test,
        'preprocessor': preprocessor
    }, 'processed_credit_data.pkl')
    print("\nSuccessfully saved processed data to processed_credit_data.pkl")

if __name__ == "__main__":
    print("Running EDA and Preprocessing...")
    
    try:
        # Load and clean data
        df = load_and_preprocess()
        
        # Generate visualizations
        df = generate_eda_plots(df)
        
        # Preprocess and save
        preprocess_data(df)
        
        print("\nProcess completed successfully. Output files:")
        print("- eda_distributions.png (EDA visualizations)")
        print("- time_analysis.png (Temporal analysis)")
        print("- processed_credit_data.pkl (Preprocessed data for modeling)")
    
    except Exception as e:
        print(f"\nError encountered: {str(e)}")
        print("\nDebugging info:")
        if 'df' in locals():
            print("\nData sample:")
            print(df.head())
            print("\nCurrent missing values:")
            print(df.isnull().sum())
            print("\nData types:")
            print(df.dtypes)
        else:
            print("Failed during initial data loading")
