import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial']
matplotlib.rcParams['font.family'] = 'sans-serif'

import sys
import argparse
from data_loader import load_data
from model_trainer import train_models, prepare_datasets
from evaluator import evaluate, save_results, plot_results
from fairness import apply_reweighting, apply_adversarial
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

def run_experiment():
    print("\n[1/4] Loading data...")
    train_data, test_data, priv, unpriv = load_data()
    
    print("\n[2/4] Training baseline models...")
    X_train, y_train = prepare_datasets(train_data)
    models = train_models(X_train, y_train)
    
    print("\n[3/4] Training fairness-aware models...")
    fair_results = {}
    
    # Reweighing
    train_reweighted = apply_reweighting(train_data, priv, unpriv)
    X_train_rw, y_train_rw = prepare_datasets(train_reweighted)
    models_rw = train_models(X_train_rw, y_train_rw)
    fair_results['XGBoost_RW'] = evaluate(test_data, models_rw['XGBoost'], priv, unpriv)
    
    # Adversarial Debiasing
    print("Training Adversarial Debiasing model...")
    adv_model = apply_adversarial(train_data, priv, unpriv)
    fair_results['AdvDebiasing'] = evaluate(test_data, adv_model, priv, unpriv)
    
    print("\n[4/4] Analyzing results...")
    all_results = {**{name: evaluate(test_data, model, priv, unpriv) 
                     for name, model in models.items()}, 
                  **fair_results}
    save_results(all_results, 'results/all_results.csv')
    plot_results('results/all_results.csv')
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['baselines', 'fairness'], default='fairness')
    args = parser.parse_args()
    
    if args.phase == 'fairness':
        success = run_experiment()
        sys.exit(0 if success else 1)