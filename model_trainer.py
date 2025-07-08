from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib

def prepare_datasets(dataset):
    X = dataset.features
    y = dataset.labels.ravel()
    return X, y  # We already split during preprocessing

def train_models(X_train, y_train):
    models = {
        'XGBoost': XGBClassifier(
            random_state=42,
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)),
        'RandomForest': RandomForestClassifier(
            random_state=42,
            class_weight='balanced')
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{name}.pkl')
        trained_models[name] = model
    
    return trained_models