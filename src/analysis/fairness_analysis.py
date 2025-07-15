import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from pathlib import Path
from src.utils import setup_logger

# Configure imports
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
from src.preprocessing.EDA import FraudDataProcessor

logger = setup_logger('fairness_logger')

def calculate_fairness_metrics(df):
    # Calculate fairness metrics for geographic and temporal dimensions
    processor = FraudDataProcessor()
    df = processor.engineer_features(df)
    
    # Table 1: Geographic analysis
    geo_metrics = (df.groupby('Location', observed=True)
                   .agg(
                       Total_Transactions=('TransactionID', 'count'),
                       Fraud_Transactions=('IsFraud', 'sum')
                   )
                   .assign(
                       Fraud_Rate_pct=lambda x: round(x['Fraud_Transactions']/x['Total_Transactions']*100, 2),
                       SPD=lambda x: x['Fraud_Rate_pct'] - x.loc[x.index == 'New York', 'Fraud_Rate_pct'].values[0]
                   ))
    
    # Table 2: Temporal analysis
    time_metrics = (df.groupby('TimeCategory', observed=True)
                     .agg(
                         Total_Transactions=('TransactionID', 'count'),
                         Fraud_Transactions=('IsFraud', 'sum')
                     )
                     .assign(
                         Fraud_Rate_pct=lambda x: round(x['Fraud_Transactions']/x['Total_Transactions']*100, 2),
                         EOD=lambda x: abs(x['Fraud_Rate_pct'] - x.loc[x.index == 'Morning', 'Fraud_Rate_pct'].values[0])
                     ))
    
    # Calculate 95% Confidence Intervals
    def calculate_ci(row):
        p = row['Fraud_Rate_pct']
        n = row['Total_Transactions']
        std_err = np.sqrt(p * (100 - p) / n)
        lower = p - 1.96 * std_err
        upper = p + 1.96 * std_err
        return f"[{lower:.1f}%, {upper:.1f}%]"
    
    time_metrics['CI_95_pct'] = time_metrics.apply(calculate_ci, axis=1)
    
    return geo_metrics, time_metrics

def save_metrics(geo_df, time_df, filename_prefix='fairness'):
    # Save fairness metrics to CSV files
    os.makedirs('results', exist_ok=True)
    geo_path = os.path.join('results', f'{filename_prefix}_geo.csv')
    time_path = os.path.join('results', f'{filename_prefix}_time.csv')
    
    geo_df.to_csv(geo_path, float_format='%.2f')
    time_df.to_csv(time_path, float_format='%.2f')
    logger.info(f"Metrics saved to {geo_path} and {time_path}")

def generate_fairness_report(geo_df, time_df):
    # Generate visual fairness report
    plt.style.use('ggplot')
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Geographic disparities
    sns.barplot(x=geo_df.index, y='SPD', data=geo_df.reset_index(), ax=ax1)
    ax1.axhline(0.1, color='r', linestyle='--', label='Fairness Threshold')
    ax1.axhline(-0.1, color='r', linestyle='--')
    ax1.set_title('Geographic Bias (SPD)')
    ax1.set_ylabel('Statistical Parity Difference')
    ax1.tick_params(axis='x', rotation=45)
    
    # Temporal disparities
    sns.barplot(x=time_df.index, y='EOD', data=time_df.reset_index(), ax=ax2)
    ax2.axhline(0.05, color='r', linestyle='--', label='Fairness Threshold')
    ax2.set_title('Temporal Bias (EOD)')
    ax2.set_ylabel('Equalized Odds Difference')
    
    plt.tight_layout()
    report_path = os.path.join('results', 'fairness_report.png')
    plt.savefig(report_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Fairness report saved to {report_path}")

def main():
    # Run fairness analysis pipeline
    logger.info("Starting Fairness Analysis...")
    
    processor = FraudDataProcessor()
    data_path = os.path.join('data', 'raw', 'credit_card_fraud_dataset.csv')
    
    # Load and process data
    df = processor.load_data(data_path)
    df = processor.engineer_features(df)
    
    # Calculate and save metrics
    geo_stats, time_stats = calculate_fairness_metrics(df)
    save_metrics(geo_stats, time_stats)
    generate_fairness_report(geo_stats, time_stats)
    
    logger.info("Analysis completed successfully")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)