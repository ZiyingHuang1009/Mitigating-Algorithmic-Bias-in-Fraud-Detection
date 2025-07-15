import os
from pathlib import Path
import argparse
from datetime import datetime
import joblib
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from ..utils import setup_logger, configure_environment
configure_environment()
logger = setup_logger()

from .data_loader import load_data
from .model_trainer import train_models
from .fairness import FairnessProcessor
from .evaluator import Evaluator

class FraudDetectionPipeline:
    def __init__(self, version='original'):
        self.results = {}
        self.models = {}
        self.logger = setup_logger()
        self.version = version
        self.fairness_processor = None
        self.base_dir = Path(__file__).parent.parent.parent
        self.results_dir = self.base_dir / 'results'
        self.results_dir.mkdir(exist_ok=True)

    def run(self, phase='fairness'):
        # 1. Load and preprocess data
        self.logger.info("Loading data...")
        train_data, test_data, priv_groups, unpriv_groups = load_data(self.version)

        # 2. Initialize components
        evaluator = Evaluator(test_data, priv_groups, unpriv_groups)
        self.fairness_processor = FairnessProcessor(priv_groups, unpriv_groups)
        
        # 3. Train baseline models
        self.logger.info("Training baseline models...")
        self.models.update(train_models(train_data))
        
        # 4. Fairness-aware training
        if phase == 'fairness':
            self.logger.info("Running fairness-aware training...")
            self._train_fairness_models(train_data)
        
        # 5. Evaluate and save results
        self.logger.info("Evaluating models...")
        self._evaluate_all(evaluator)
        results_path = self._save_results()
        
        return self.results, str(results_path)

    def _train_fairness_models(self, train_data):
        # Handle fairness training with proper validation
        rw_data = self.fairness_processor.apply_reweighting(train_data)
        if rw_data is None:
            self.logger.warning("Using original data for fairness training (reweighting failed)")
            rw_data = train_data
            
        trained_models = train_models(rw_data)
        if not trained_models:
            raise ValueError("No models were successfully trained")
            
        # Only store successfully trained models
        for name, model in trained_models.items():
            if model is not None:
                self.models[f"{name}_RW"] = model

    def _evaluate_all(self, evaluator):
        # Evaluate only successful models
        for name, model in list(self.models.items()):
            if model is None:
                self.logger.error(f"Skipping evaluation for {name} - model is None")
                del self.models[name]
                continue
                
            if 'AdvDebiasing' in name:
                with self.fairness_processor.active_session():
                    self.results[name] = evaluator.evaluate_adversarial(model)
            else:
                self.results[name] = evaluator.evaluate(model)

    def _save_results(self):
        # Save results to CSV with validation
        results_file = self.results_dir / f'fairness_metrics_{self.version}.csv'
        import pandas as pd
        df = pd.DataFrame.from_dict(self.results, orient='index')
        df.to_csv(results_file, float_format='%.4f')
        
        if not results_file.exists():
            raise FileNotFoundError(f"Results file not created at {results_file}")
        
        self.logger.info(f"Results successfully saved to {results_file}")
        return results_file

    def _save_models(self):
        # Save models with validation
        model_dir = self.base_dir / 'models'
        model_dir.mkdir(exist_ok=True)
        
        for name, model in self.models.items():
            if 'AdvDebiasing' not in name and model is not None:
                model_file = model_dir / f'{name}_{self.version}.pkl'
                joblib.dump(model, model_file)
                self.logger.info(f"Saved model: {model_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fraud Detection Pipeline')
    parser.add_argument('--phase', choices=['baselines', 'fairness'], default='fairness')
    parser.add_argument('--version', type=str, default='original', help='Data version to use')
    args = parser.parse_args()
    
    pipeline = FraudDetectionPipeline(version=args.version)
    results, results_path = pipeline.run(args.phase)
    
    print("\nFinal Results:")
    print(results)
    print(f"\nResults saved to: {results_path}")