import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aif360.metrics import ClassificationMetric
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, confusion_matrix,
    average_precision_score, precision_recall_curve, classification_report
)
import numpy as np
import os
import logging
from typing import Dict, Any, Optional

class Evaluator:
    def __init__(self, test_data, privileged_groups, unprivileged_groups):
        # Initialize evaluator with validation
        if not hasattr(test_data, 'features') or not hasattr(test_data, 'labels'):
            raise ValueError("Test data must have features and labels")
        if not privileged_groups or not unprivileged_groups:
            raise ValueError("Both privileged and unprivileged groups must be specified")
        
        self.test_data = test_data
        self.priv = privileged_groups
        self.unpriv = unprivileged_groups
        self.logger = logging.getLogger(__name__)
        os.makedirs('results', exist_ok=True)

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        # Calculate comprehensive evaluation metrics
        if len(y_true) != len(y_pred):
            raise ValueError("Length mismatch between y_true and y_pred")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob) if y_prob is not None and len(np.unique(y_true)) > 1 else np.nan,
            'pr_auc': average_precision_score(y_true, y_prob) if y_prob is not None else np.nan,
            'fraud_capture_rate': self._calculate_fraud_capture_rate(y_true, y_pred),
            'false_positive_rate': self._calculate_false_positive_rate(y_true, y_pred)
        }

        # Add fairness metrics if protected attributes are valid
        if (hasattr(self.test_data, 'protected_attributes') and 
            len(np.unique(self.test_data.protected_attributes)) >= 2):
            
            dataset_pred = self.test_data.copy()
            dataset_pred.labels = y_pred.reshape(-1, 1)
            
            fair_metrics = ClassificationMetric(
                self.test_data, dataset_pred,
                unprivileged_groups=self.unpriv,
                privileged_groups=self.priv
            )
            
            metrics.update({
                'statistical_parity_diff': self._safe_metric(fair_metrics.statistical_parity_difference),
                'equal_opp_diff': self._safe_metric(fair_metrics.equal_opportunity_difference),
                'disparate_impact': self._safe_metric(fair_metrics.disparate_impact),
                'average_odds_diff': self._safe_metric(fair_metrics.average_odds_difference),
                'false_positive_rate_diff': self._safe_metric(fair_metrics.false_positive_rate_difference)
            })
        else:
            metrics.update({
                'statistical_parity_diff': np.nan,
                'equal_opp_diff': np.nan,
                'disparate_impact': np.nan,
                'average_odds_diff': np.nan,
                'false_positive_rate_diff': np.nan
            })
        
        return metrics

    def _safe_metric(self, metric_func) -> float:
        # Safe wrapper for fairness metrics
        with np.errstate(divide='ignore', invalid='ignore'):
            result = metric_func()
            return float(result) if np.isfinite(result) else np.nan

    def _calculate_fraud_capture_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate percentage of actual fraud cases caught
        fraud_idx = y_true == 1
        if sum(fraud_idx) == 0:
            return np.nan
        return float(np.sum(y_pred[fraud_idx] == 1) / np.sum(fraud_idx))

    def _calculate_false_positive_rate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Calculate false positive rate
        non_fraud_idx = y_true == 0
        if sum(non_fraud_idx) == 0:
            return np.nan
        return float(np.sum(y_pred[non_fraud_idx] == 1) / np.sum(non_fraud_idx))

    def evaluate(self, model, model_name: Optional[str] = None) -> Dict[str, Any]:
        # Evaluate model with comprehensive metrics
        X = self.test_data.features
        y_true = self.test_data.labels.ravel()
        
        # Get predictions and probabilities
        y_prob = None
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X)[:, 1]
            y_pred = (y_prob > self._find_optimal_threshold(y_true, y_prob)).astype(int)
        else:
            y_pred = model.predict(X)
        
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        
        # Save detailed classification report
        if model_name:
            report = classification_report(y_true, y_pred, output_dict=True)
            pd.DataFrame(report).transpose().to_csv(
                f"results/{model_name}_classification_report.csv"
            )
        
        return metrics

    def _find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        # Find optimal threshold based on precision-recall tradeoff
        if len(np.unique(y_true)) < 2:
            return 0.5
            
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        return thresholds[np.nanargmax(f1_scores)]

    def evaluate_adversarial(self, model, dataset=None) -> Dict[str, Any]:
        # Special evaluation for adversarial models
        dataset = dataset or self.test_data
        pred_data = model.predict(dataset)
        y_true = dataset.labels.ravel()
        y_pred = pred_data.labels.ravel()
        return self._calculate_metrics(y_true, y_pred)

    def save_results(self, results: Dict[str, Any], path: str) -> None:
        # Save results with validation
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(path, float_format='%.4f')
        self.logger.info(f"Results successfully saved to {os.path.abspath(path)}")

    def plot_confusion_matrices(self, models: Dict[str, Any]) -> None:
        # Enhanced confusion matrix visualization
        n_models = len(models)
        fig, axes = plt.subplots(
            nrows=(n_models + 1) // 2, 
            ncols=min(2, n_models),
            figsize=(15, 5 * ((n_models + 1) // 2))
        )
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for i, (name, model) in enumerate(models.items()):
            ax = axes[i]
            
            if 'AdvDebiasing' in name:
                preds = model.predict(self.test_data).labels.ravel()
            else:
                X = self.test_data.features
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(X)[:, 1]
                    preds = (y_prob > self._find_optimal_threshold(self.test_data.labels.ravel(), y_prob)).astype(int)
                else:
                    preds = model.predict(X)
            
            cm = confusion_matrix(self.test_data.labels.ravel(), preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Non-Fraud', 'Fraud'],
                       yticklabels=['Non-Fraud', 'Fraud'])
            
            recall = recall_score(self.test_data.labels.ravel(), preds)
            precision = precision_score(self.test_data.labels.ravel(), preds)
            fpr = self._calculate_false_positive_rate(self.test_data.labels.ravel(), preds)
            
            ax.set_title(
                f'{name}\n'
                f'Recall: {recall:.2f} | Precision: {precision:.2f}\n'
                f'FPR: {fpr:.2f}'
            )
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        plt.tight_layout()
        output_path = os.path.join('results', 'confusion_matrices.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Confusion matrices saved to {output_path}")

    def plot_precision_recall_curves(self, models: Dict[str, Any]) -> None:
        # Plot precision-recall curves for all models
        plt.figure(figsize=(10, 8))
        y_true = self.test_data.labels.ravel()
        
        for name, model in models.items():
            if hasattr(model, 'predict_proba'):
                X = self.test_data.features
                y_prob = model.predict_proba(X)[:, 1]
                precision, recall, _ = precision_recall_curve(y_true, y_prob)
                plt.plot(recall, precision, lw=2, label=f'{name} (AP={average_precision_score(y_true, y_prob):.2f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='best')
        plt.grid(True)
        
        output_path = os.path.join('results', 'precision_recall_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Precision-Recall curves saved to {output_path}")