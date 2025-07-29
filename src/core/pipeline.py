from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
import pandas as pd
from datetime import datetime
from pathlib import Path
import logging
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate_difference
)
import gc
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional, Union
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, balanced_accuracy_score, roc_curve
)
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class Pipeline:
    # Optimized fraud detection pipeline with balanced performance metrics.
    
    def __init__(self, version: str = "smote") -> None:
        self.version = version
        self.logger = logging.getLogger('pipeline')
        self._setup_directories()
        
        # Configuration
        self.max_sample_size = 30000
        self.safe_mode = True
        self.protected_attrs: Optional[List[str]] = None
        self.pos_class_weight = 15  # Balanced weight for fraud class

    def _setup_directories(self) -> None:
        self.results_dir = Path("results") / self.version
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def _add_features(self, X: pd.DataFrame) -> pd.DataFrame:
        # Optimized feature engineering focusing on fraud indicators.
        X = X.copy()
        
        # Temporal patterns
        if 'Timestamp' in X.columns:
            X['hour'] = X['Timestamp'].dt.hour
            X['is_night'] = (X['hour'] < 6) | (X['hour'] > 20)
        
        # Transaction patterns
        if 'Amount' in X.columns:
            X['log_amount'] = np.log1p(X['Amount'])
            X['large_amount'] = (X['Amount'] > X['Amount'].quantile(0.95)).astype(int)
        
        # User behavior
        if 'UserID' in X.columns:
            user_stats = X.groupby('UserID')['Amount'].agg(['mean', 'std']).add_prefix('user_amt_')
            X = X.join(user_stats, on='UserID')
            X['amt_deviation'] = (X['Amount'] - X['user_amt_mean']) / (X['user_amt_std'] + 1e-6)
        
        return X

    def _initialize_models(self) -> Dict[str, Dict[str, BaseEstimator]]:
        # Initialize models using the enhanced ModelTrainer
        from src.core.model_trainer import ModelTrainer
        return {
            'baseline': {
                'XGBoost_Fraud': XGBClassifier(
                    random_state=42,
                    scale_pos_weight=self.pos_class_weight,
                    eval_metric='aucpr',
                    max_depth=3,  # Reduced from 5
                    learning_rate=0.05,  # Reduced from 0.1
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=1,
                    reg_lambda=1,
                    n_estimators=300  # Increased from 150
                ),
                'BalancedForest': BalancedRandomForestClassifier(
                    random_state=42,
                    n_estimators=150,  # Increased from 100
                    max_depth=5,  # Reduced from 8
                    sampling_strategy='all',
                    class_weight='balanced_subsample',
                    n_jobs=-1
                )
            }
        }

    def _train_with_cv(self, X_train, y_train, X_test, y_test):
        # Optimized training with focus on fraud recall/precision balance.
        models = self._initialize_models()
        results = {}
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        for name, model in tqdm(models['baseline'].items(), desc="Training models"):
            start_time = time.time()
            val_recalls = []
            test_preds = []
            test_probs = []
            
            # Check if we have valid data
            if X_train.empty or y_train.empty:
                self.logger.error(f"Empty training data for {name}")
                results[name] = {'error': 'Empty training data'}
                continue
                
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                if 'XGBoost' in name:
                    model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(X_tr, y_tr)
                
                # Validate recall
                val_pred = model.predict(X_val)
                val_recalls.append(recall_score(y_val, val_pred, pos_label=1))
                
                # Test predictions
                test_preds.append(model.predict(X_test))
                if hasattr(model, 'predict_proba'):
                    test_probs.append(model.predict_proba(X_test)[:, 1])
            
            # Ensemble predictions
            y_pred = np.round(np.mean(test_preds, axis=0)).astype(int)
            y_probs = np.mean(test_probs, axis=0) if test_probs else None
            
            # Calculate metrics with fraud focus
            metrics = self._calculate_fraud_metrics(y_test, y_pred, X_test, y_probs)
            metrics.update({
                'training_time': time.time() - start_time,
                'avg_val_recall': np.mean(val_recalls)
            })
            
            results[name] = metrics
            self._save_model_plots(model, X_test, y_test, name, y_probs)
            
            gc.collect()
        
        return results

    def _calculate_fraud_metrics(self, y_true, y_pred, X, y_probs=None):
        # Metrics focused on fraud detection performance.
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        total_pos = tp + fn
        
        metrics = {
            'fraud_precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
            'fraud_recall': recall_score(y_true, y_pred, pos_label=1, zero_division=0),
            'fraud_f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            'fraud_capture_rate': tp / total_pos if total_pos > 0 else 0,
            'false_positives': fp,
            'false_negatives': fn,
            'roc_auc': roc_auc_score(y_true, y_probs) if y_probs is not None else np.nan,
            'pr_auc': average_precision_score(y_true, y_probs) if y_probs is not None else np.nan,
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
        }
        
        # Fairness metrics
        if self.protected_attrs:
            sensitive_features = X[self.protected_attrs].values
            metrics.update({
                'demographic_parity_diff': demographic_parity_difference(
                    y_true, y_pred, sensitive_features=sensitive_features),
                'equalized_odds_diff': equalized_odds_difference(
                    y_true, y_pred, sensitive_features=sensitive_features)
            })
        
        return metrics

    def _save_model_plots(self, model, X_test, y_test, model_name, y_probs):
        # Save diagnostic plots for model evaluation.
        if y_probs is None:
            self.logger.warning(f"No probabilities available for {model_name}")
            return
            
        # Debugging output
        self.logger.debug(f"Plotting debug - {model_name}:")
        self.logger.debug(f"y_test shape: {y_test.shape}")
        self.logger.debug(f"y_probs shape: {y_probs.shape}")
        self.logger.debug(f"Unique y_test values: {np.unique(y_test)}")
        
        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision)
        plt.title(f'{model_name} Precision-Recall Curve (AP={average_precision_score(y_test, y_probs):.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(self.results_dir / f'pr_curve_{model_name}.png', dpi=300)
        plt.close()
        
        # Threshold Optimization
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, gmeans[:-1])  # Note: gmeans has one more element than thresholds
        plt.axvline(thresholds[ix], color='r', linestyle='--', 
                   label=f'Optimal: {thresholds[ix]:.2f}')
        plt.title(f'{model_name} Threshold Optimization (G-Mean={gmeans[ix]:.2f})')
        plt.xlabel('Threshold')
        plt.ylabel('G-Mean Score')
        plt.legend()
        plt.savefig(self.results_dir / f'threshold_opt_{model_name}.png', dpi=300)
        plt.close()

    def run_evaluation(self, train_data, test_data, priv_groups, unpriv_groups):
        # Run optimized evaluation pipeline.
        X_train, y_train = train_data
        X_test, y_test = test_data
        
        # Feature engineering
        X_train_fe = self._add_features(X_train)
        X_test_fe = self._add_features(X_test)
        
        # Model training
        results = self._train_with_cv(X_train_fe, y_train, X_test_fe, y_test)
        
        return self._save_results(results, "model_comparison")

    def _save_results(self, results, prefix):
        # Save optimized results format.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"{prefix}_{timestamp}.csv"
        pd.DataFrame.from_dict(results, orient='index').to_csv(results_path)
        return results, results_path

    def run(self, phase: str):
        # Execute pipeline phase.
        from src.core.data_loader import DataLoader
        
        train_data, test_data, priv_groups, unpriv_groups = DataLoader(self.version).load_data()
        self.protected_attrs = list({attr for group in priv_groups for attr in group.keys()})
        
        if phase == "evaluation":
            return self.run_evaluation(train_data, test_data, priv_groups, unpriv_groups)
        elif phase == "fairness":
            return self.run_fairness_analysis(train_data, test_data, priv_groups, unpriv_groups)
        
        self.logger.error(f"Invalid phase: {phase}")
        raise ValueError(f"Invalid phase: {phase}")