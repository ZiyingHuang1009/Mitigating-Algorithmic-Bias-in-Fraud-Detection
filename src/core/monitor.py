import pandas as pd
from alibi_detect import TabularDrift

class FraudMonitor:
    def __init__(self, X_train):
        self.drift_detector = TabularDrift(
            X_train.values,
            p_val=0.05
        )
    
    def check_drift(self, X_new):
        return self.drift_detector.predict(X_new.values)
    
    def track_performance(self, y_true, y_pred, window=1000):
        return pd.DataFrame({
            'accuracy': y_true.rolling(window).apply(lambda x: (x == y_pred[x.index]).mean()),
            'fraud_rate': y_pred.rolling(window).mean()
        })