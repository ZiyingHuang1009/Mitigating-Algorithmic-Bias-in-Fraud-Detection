import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, recall_score
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
import seaborn as sns
import matplotlib.style as mplstyle

# Configuration
RANDOM_STATE = 42
VALID_STYLES = plt.style.available

def setup_plot_style():
    # Set up consistent plot style with validation
    style = 'seaborn-v0_8' if 'seaborn-v0_8' in VALID_STYLES else 'ggplot'
    plt.style.use(style)
    sns.set_palette("husl")
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.size': 12
    })

def validate_data_files():
    required_files = [
        'data/processed/X_train_smote.csv',
        'data/processed/y_train_smote.csv',
        'data/processed/X_test.csv',
        'data/processed/y_test.csv'
    ]
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data files: {missing}")

def load_data():
    # Load and validate preprocessed data
    validate_data_files()
    return (
        pd.read_csv('data/processed/X_train_smote.csv'),
        pd.read_csv('data/processed/y_train_smote.csv').squeeze(),
        pd.read_csv('data/processed/X_test.csv'),
        pd.read_csv('data/processed/y_test.csv').squeeze()
    )

def detect_sensitive_features(X):
    # Find and validate geographic and temporal features
    geo_features = [col for col in X.columns if col.startswith('Location_')]
    time_features = [col for col in X.columns if col.startswith('Time_')]
    
    if not geo_features:
        raise ValueError("No geographic features found (expected columns starting with 'Location_')")
    if not time_features:
        raise ValueError("No temporal features found (expected columns starting with 'Time_')")
    
    return geo_features, time_features

def train_model(X_train, y_train):
    # Train and validate random forest model
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_estimators=100,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def generate_confusion_matrix(y_true, y_pred):
    # Generate and save confusion matrix
    Path('results').mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'],
                ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    fig.savefig('results/confusion_matrices.png', bbox_inches='tight')
    plt.close(fig)

def calculate_group_metrics(y_true, y_pred, group_mask, group_name):
    # Calculate metrics for a specific group
    group_y_true = y_true[group_mask]
    group_y_pred = y_pred[group_mask]
    
    return {
        'group': group_name,
        'demographic_parity_diff': demographic_parity_difference(
            y_true, y_pred, 
            sensitive_features=group_mask.astype(int)
        ),
        'equalized_odds_diff': equalized_odds_difference(
            y_true, y_pred,
            sensitive_features=group_mask.astype(int)
        ),
        'fraud_rate': group_y_pred.mean(),
        'fraud_recall': recall_score(group_y_true, group_y_pred, pos_label=1, zero_division=0)
    }

def calculate_fairness_metrics(X_test, y_test, y_pred, sensitive_features, prefix):
    # Calculate and save fairness metrics by sensitive group
    group_metrics = [
        calculate_group_metrics(
            y_test, y_pred,
            (X_test[feature] == group),
            f"{feature}_{group}"
        )
        for feature in sensitive_features
        for group in X_test[feature].unique()
    ]
    
    metrics_df = pd.DataFrame(group_metrics)
    metrics_df.to_csv(f'results/fairness_{prefix}.csv', index=False)
    return metrics_df

def generate_fairness_report(geo_metrics, time_metrics):
    # Generate combined fairness visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Geographic fairness plot
    geo_metrics.plot.bar(
        x='group', y='demographic_parity_diff', 
        ax=ax1, color='skyblue'
    )
    ax1.set_title('Geographic Fairness (Demographic Parity)')
    ax1.set_ylabel('Difference')
    ax1.tick_params(axis='x', rotation=45)
    
    # Temporal fairness plot
    time_metrics.plot.bar(
        x='group', y='demographic_parity_diff', 
        ax=ax2, color='lightgreen'
    )
    ax2.set_title('Temporal Fairness (Demographic Parity)')
    ax2.set_ylabel('Difference')
    ax2.tick_params(axis='x', rotation=45)
    
    fig.tight_layout()
    fig.savefig('results/fairness_report.png', dpi=300)
    plt.close(fig)

def run_analysis():
    # Main analysis pipeline with explicit validation
    setup_plot_style()
    
    # Load and validate data
    X_train, y_train, X_test, y_test = load_data()
    geo_features, time_features = detect_sensitive_features(X_test)
    
    print(f"Analyzing {len(X_test)} transactions with features:")
    print(f"- Geographic: {geo_features[:3]}...")
    print(f"- Temporal: {time_features[:3]}...")
    
    # Train model
    model = train_model(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Generate outputs
    generate_confusion_matrix(y_test, y_pred)
    
    geo_metrics = calculate_fairness_metrics(
        X_test, y_test, y_pred, geo_features, 'geo'
    )
    time_metrics = calculate_fairness_metrics(
        X_test, y_test, y_pred, time_features, 'time'
    )
    
    generate_fairness_report(geo_metrics, time_metrics)
    
    print("\nAnalysis complete. Generated files:")
    print("- results/confusion_matrices.png")
    print("- results/fairness_geo.csv")
    print("- results/fairness_time.csv")
    print("- results/fairness_report.png")

if __name__ == '__main__':
    run_analysis()
