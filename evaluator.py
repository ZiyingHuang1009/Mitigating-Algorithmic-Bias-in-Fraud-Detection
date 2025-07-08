import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from aif360.metrics import ClassificationMetric
from sklearn.metrics import (accuracy_score, f1_score, 
                           precision_score, recall_score,
                           roc_auc_score)

def evaluate(dataset, model, privileged_groups, unprivileged_groups):
    """Evaluate model performance and fairness metrics"""
    X = dataset.features
    y_true = dataset.labels.ravel()
    
    try:
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    except AttributeError:
        # For adversarial debiasing models
        y_pred = dataset.labels
        y_prob = None

    # Create dataset with predictions
    dataset_pred = dataset.copy()
    dataset_pred.labels = y_pred.reshape(-1, 1)
    
    # Performance metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    }
    
    # Fairness metrics
    classified_metric = ClassificationMetric(
        dataset, dataset_pred,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups)
    
    metrics.update({
        'statistical_parity_diff': classified_metric.statistical_parity_difference(),
        'equal_opp_diff': classified_metric.equal_opportunity_difference(),
        'disparate_impact': classified_metric.disparate_impact(),
        'average_odds_diff': classified_metric.average_odds_difference()
    })
    
    return metrics

def save_results(results, path):
    """Save evaluation results to CSV"""
    df = pd.DataFrame.from_dict(results, orient='index')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Results saved to {path}")
    return df

def plot_results(results_path):
    """Generate comprehensive visualizations of results"""
    df = pd.read_csv(results_path, index_col=0)
    os.makedirs('results', exist_ok=True)
    
    # 1. Performance Metrics
    plt.figure(figsize=(12, 6))
    df[['accuracy', 'f1', 'precision', 'recall']].plot(kind='bar')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/performance_metrics.png')
    plt.close()
    
    # 2. Fairness Metrics
    fairness_metrics = ['statistical_parity_diff', 'equal_opp_diff', 'average_odds_diff']
    
    plt.figure(figsize=(12, 6))
    df[fairness_metrics].plot(kind='bar')
    plt.title('Fairness Metrics Comparison')
    plt.ylabel('Difference')
    plt.axhline(0, color='black', linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/fairness_metrics.png')
    plt.close()
    
    # 3. Trade-off Analysis
    plt.figure(figsize=(10, 6))
    for model in df.index:
        plt.scatter(
            df.loc[model, 'accuracy'], 
            abs(df.loc[model, 'statistical_parity_diff']),
            s=200, label=model
        )
    plt.xlabel('Accuracy')
    plt.ylabel('Absolute Statistical Parity Difference')
    plt.title('Accuracy vs Fairness Trade-off')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/tradeoff_analysis.png')
    plt.close()