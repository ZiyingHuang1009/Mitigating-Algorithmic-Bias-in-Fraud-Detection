import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score
import matplotlib.pyplot as plt

def optimize_threshold(model, X, y_true, pred_scores=None):
    if pred_scores is None:
        y_probs = model.predict_proba(X)[:,1]
    else:
        y_probs = pred_scores
        
    y_probs = model.predict_proba(X)[:,1]
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Calculate F1 scores
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    # Find optimal threshold
    optimal_idx = np.argmax(f1_scores[:-1])
    optimal_threshold = thresholds[optimal_idx]
    
    # Visualization
    plt.figure(figsize=(10,6))
    plt.plot(thresholds, f1_scores[:-1], label='F1-score')
    plt.axvline(optimal_threshold, color='r', linestyle='--', 
               label=f'Optimal Threshold ({optimal_threshold:.3f})')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Optimization')
    plt.legend()
    plt.savefig('threshold_optimization.png')
    plt.close()
    
    return optimal_threshold