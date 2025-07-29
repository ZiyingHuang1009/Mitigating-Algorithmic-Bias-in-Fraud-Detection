from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
import pandas as pd
from ..evaluation.threshold_opt import optimize_threshold
from ..utils import setup_logger
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                           roc_auc_score, average_precision_score, confusion_matrix)
from collections import defaultdict
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

logger = setup_logger('model_trainer')

class ModelTrainer:
    def __init__(self, version='smote'):
        self.version = version
        self.best_thresholds = {}
        self.feature_importances = defaultdict(dict)

    def train_models(self, X_train, y_train, X_test, y_test, sensitive_features=None):
        # Enhanced model training with fraud-specific optimizations
        results = {}
        
        # Validate input data
        if not self._validate_input_data(X_train, y_train, X_test, y_test):
            return pd.DataFrame()

        # Initialize optimized models
        models = {
            'XGBoost_Fraud': self._get_xgboost_model(y_train),
            'BalancedForest': self._get_balanced_forest_model(),
            'IsolationForest': self._get_isolation_forest_model()
        }

        for name, model in models.items():
            start_time = time.time()
            metrics = None
            
            if name == 'IsolationForest':
                metrics = self._train_anomaly_model(model, X_train, X_test, y_test)
            else:
                metrics = self._train_standard_model(model, X_train, y_train, X_test, y_test, sensitive_features)
            
            if metrics:
                metrics['training_time'] = time.time() - start_time
                results[name] = metrics
            else:
                logger.error(f"Model {name} failed to produce metrics")
                results[name] = {'error': 'Model training failed'}
        
        return pd.DataFrame.from_dict(results, orient='index')

    def _validate_input_data(self, X_train, y_train, X_test, y_test):
        # Validate all input data before processing
        if X_train.empty or X_test.empty:
            logger.error("Empty training or test data provided")
            return False
            
        if len(y_train) == 0 or len(y_test) == 0:
            logger.error("Empty labels provided")
            return False
            
        if len(X_train) != len(y_train):
            logger.error(f"Training data shape mismatch: X({len(X_train)}) vs y({len(y_train)})")
            return False
            
        if len(X_test) != len(y_test):
            logger.error(f"Test data shape mismatch: X({len(X_test)}) vs y({len(y_test)})")
            return False
            
        return True

    def _train_standard_model(self, model, X_train, y_train, X_test, y_test, sensitive_features):
        # Train and evaluate standard classification models
        if not self._fit_model(model, X_train, y_train):
            return None
            
        self._store_feature_importances(model.__class__.__name__, model, X_train)
        
        optimal_threshold = optimize_threshold(model, X_test, y_test)
        if optimal_threshold is None:
            logger.warning("Threshold optimization failed, using default 0.5")
            optimal_threshold = 0.5
            
        self.best_thresholds[model.__class__.__name__] = optimal_threshold
        
        y_pred = self._predict_with_threshold(model, X_test, optimal_threshold)
        if y_pred is None:
            return None
            
        return self._calculate_fraud_metrics(y_test, y_pred, sensitive_features)

    def _fit_model(self, model, X, y):
        # Safely fit a model with validation
        if not hasattr(model, 'fit'):
            logger.error("Model object has no fit method")
            return False
            
        model.fit(X, y)
        return True

    def _predict_with_threshold(self, model, X, threshold):
        # Make predictions with threshold validation
        if not hasattr(model, 'predict_proba'):
            logger.warning("Model doesn't support predict_proba, using direct prediction")
            return model.predict(X)
            
        probas = model.predict_proba(X)
        if probas.shape[1] < 2:
            logger.warning("Model returned single class probabilities")
            return (probas >= threshold).astype(int)
            
        return (probas[:,1] >= threshold).astype(int)

    def _train_anomaly_model(self, model, X_train, X_test, y_test):
        # Special handling for anomaly detection models
        if not hasattr(model, 'fit') or not hasattr(model, 'decision_function'):
            logger.error("Invalid anomaly detection model")
            return None
            
        model.fit(X_train)
        scores = -model.decision_function(X_test)  # Convert to positive = anomaly
        
        optimal_threshold = optimize_threshold(None, X_test, y_test, pred_scores=scores)
        if optimal_threshold is None:
            logger.warning("Threshold optimization failed for anomaly model")
            optimal_threshold = 0.5
            
        y_pred = (scores >= optimal_threshold).astype(int)
        return self._calculate_fraud_metrics(y_test, y_pred)

    def _get_xgboost_model(self, y_train):
        # Optimized XGBoost configuration for fraud detection
        return XGBClassifier(
            scale_pos_weight=self._calculate_class_weight(y_train),
            eval_metric='aucpr',
            tree_method='hist',
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=300,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42
        )

    def _get_balanced_forest_model(self):
        # Enhanced balanced random forest
        return BalancedRandomForestClassifier(
            n_estimators=150,
            max_depth=5,
            sampling_strategy='all',
            class_weight='balanced_subsample',
            n_jobs=-1,
            random_state=42
        )

    def _get_isolation_forest_model(self):
        # Anomaly detection approach
        return IsolationForest(
            n_estimators=200,
            contamination='auto',
            random_state=42,
            n_jobs=-1
        )

    def _calculate_class_weight(self, y):
        # Dynamic class weight calculation
        class_counts = np.bincount(y)
        if len(class_counts) < 2:
            logger.warning("Only one class present, using default class weight")
            return 1.0
        return min(10, class_counts[0] / class_counts[1])

    def _calculate_fraud_metrics(self, y_true, y_pred, sensitive_features=None):
        # Comprehensive fraud-specific metrics with validation
        if len(y_true) != len(y_pred):
            logger.error(f"Metric calculation mismatch: y_true({len(y_true)}) vs y_pred({len(y_pred)})")
            return None
            
        cm = confusion_matrix(y_true, y_pred)
        if cm.size != 4:
            logger.error("Confusion matrix calculation failed")
            return None
            
        tn, fp, fn, tp = cm.ravel()
        total_pos = tp + fn
        
        metrics = {
            'fraud_precision': precision_score(y_true, y_pred, zero_division=0),
            'fraud_recall': recall_score(y_true, y_pred, zero_division=0),
            'fraud_f1': f1_score(y_true, y_pred, zero_division=0),
            'fraud_capture_rate': tp / total_pos if total_pos > 0 else 0,
            'false_positives': fp,
            'false_negatives': fn,
            'roc_auc': roc_auc_score(y_true, y_pred),
            'pr_auc': average_precision_score(y_true, y_pred)
        }
        
        if sensitive_features is not None and hasattr(self, '_calculate_fairness_metrics'):
            metrics.update(self._calculate_fairness_metrics(y_true, y_pred, sensitive_features))
            
        return metrics

    def _store_feature_importances(self, model_name, model, X_train):
        # Capture and store feature importance data with validation
        if not hasattr(model, 'feature_importances_') and not hasattr(model, 'coef_'):
            logger.debug(f"No feature importance available for {model_name}")
            return
            
        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.abs(model.coef_[0])
        self.feature_importances[model_name] = dict(zip(X_train.columns, importances))
