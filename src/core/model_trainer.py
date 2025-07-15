from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from ..utils import setup_logger
from sklearn.utils.class_weight import compute_class_weight

logger = setup_logger('model_trainer')

def calculate_class_weight(y: np.ndarray) -> float:
    # Calculate class weight ratio for imbalanced datasets with validation
    if not isinstance(y, np.ndarray):
        raise TypeError("Input y must be a numpy array")
    
    classes = np.unique(y)
    if len(classes) < 2:
        logger.warning("Only one class found in labels, using neutral weight")
        return 1.0
    
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return min(weights[1] / weights[0], 100)  # Cap at 100 to prevent extreme weights

def train_models(train_data, val_data=None):
    # Train machine learning models with explicit validation
    # Validate input data structure
    if not hasattr(train_data, 'features') or not hasattr(train_data, 'labels'):
        raise ValueError("Training data must have 'features' and 'labels'")
    
    if val_data and (not hasattr(val_data, 'features') or not hasattr(val_data, 'labels')):
        raise ValueError("Validation data must have 'features' and 'labels'")

    X_train = train_data.features
    y_train = train_data.labels.ravel()
    
    # Validate shapes
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Feature/label shape mismatch: {X_train.shape[0]} samples vs {y_train.shape[0]} labels")
    
    logger.info(f"Training data shapes - Features: {X_train.shape}, Labels: {y_train.shape}")
    
    # Calculate class weights
    class_weight_ratio = calculate_class_weight(y_train)
    logger.info(f"Class weight ratio: {class_weight_ratio:.2f}")
    
    # Model configurations with validation
    models = {
        'XGBoost': XGBClassifier(
            scale_pos_weight=class_weight_ratio,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            eval_metric=['aucpr', 'error'],
            early_stopping_rounds=10 if val_data else None,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            class_weight='balanced_subsample',
            max_depth=7,
            min_samples_leaf=10,
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
    }
    
    trained = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        
        # XGBoost specific validation handling
        if name == 'XGBoost' and val_data:
            X_val = val_data.features
            y_val = val_data.labels.ravel()
            
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError(f"Validation feature dimension mismatch: {X_val.shape[1]} vs {X_train.shape[1]}")
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=True
            )
        else:
            model.fit(X_train, y_train)
        
        trained[name] = model
        logger.info(f"{name} trained successfully")
    
    return trained