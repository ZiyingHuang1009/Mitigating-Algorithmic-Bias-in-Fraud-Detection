import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from EDA import load_and_preprocess  # Reuse data loading from EDA.py

def calculate_fairness_metrics(df):
    # Ensure TimeCategory exists
    if 'TimeCategory' not in df:
        df['TimeCategory'] = pd.cut(df['TransactionHour'],
                                  bins=[0, 6, 12, 18, 24],
                                  labels=['Night','Morning','Afternoon','Evening'])
    
    # Table 1: Geographic analysis
    geo_metrics = (df.groupby('Location')
                   .agg(
                       Total_Transactions=('TransactionID', 'count'),
                       Fraud_Transactions=('IsFraud', 'sum')
                   )
                   .assign(
                       Fraud_Rate_pct=lambda x: round(x['Fraud_Transactions']/x['Total_Transactions']*100, 2),
                       SPD=lambda x: x['Fraud_Rate_pct'] - x.loc[x.index == 'New York', 'Fraud_Rate_pct'].values[0]
                   ))
    
    # Table 2: Temporal analysis
    time_metrics = (df.groupby('TimeCategory')
                     .agg(
                         Total_Transactions=('TransactionID', 'count'),
                         Fraud_Transactions=('IsFraud', 'sum')
                     )
                     .assign(
                         Fraud_Rate_pct=lambda x: round(x['Fraud_Transactions']/x['Total_Transactions']*100, 2),
                         EOD=lambda x: abs(x['Fraud_Rate_pct'] - x.loc[x.index == 'Morning', 'Fraud_Rate_pct'].values[0])
                     ))
    
    # Calculate 95% Confidence Intervals
    time_metrics['CI_95_pct'] = time_metrics.apply(
        lambda row: f"[{row['Fraud_Rate_pct']-1.96*np.sqrt(row['Fraud_Rate_pct']*(100-row['Fraud_Rate_pct'])/row['Total_Transactions']):.1f}%, "
                   f"{row['Fraud_Rate_pct']+1.96*np.sqrt(row['Fraud_Rate_pct']*(100-row['Fraud_Rate_pct'])/row['Total_Transactions']):.1f}%]",
        axis=1
    )
    
    return geo_metrics, time_metrics

def save_metrics(geo_df, time_df, filename_prefix='fairness'):
    # Save geographic metrics
    geo_df.to_csv(f'{filename_prefix}_geo.csv', 
                 encoding='utf-8-sig',
                 float_format='%.2f')
    
    # Save temporal metrics
    time_df.to_csv(f'{filename_prefix}_time.csv', 
                  encoding='utf-8-sig',
                  float_format='%.2f')

def generate_fairness_report(geo_df, time_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Geographic disparities plot
    sns.barplot(x=geo_df.index, y='SPD', data=geo_df.reset_index(), ax=ax1)
    ax1.axhline(0.1, color='r', linestyle='--', label='Fairness Threshold')
    ax1.axhline(-0.1, color='r', linestyle='--')
    ax1.set_title('Geographic Bias (Statistical Parity Difference)')
    ax1.set_ylabel('SPD (Percentage Points)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()
    
    # Temporal disparities plot
    sns.barplot(x=time_df.index, y='EOD', data=time_df.reset_index(), ax=ax2)
    ax2.axhline(0.05, color='r', linestyle='--', label='Fairness Threshold')
    ax2.set_title('Temporal Bias (Equalized Odds Difference)')
    ax2.set_ylabel('EOD (Percentage Points)')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('fairness_report.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Running Fairness Analysis...")
    
    # Load and preprocess data
    transaction_data = load_and_preprocess()
    
    # Calculate fairness metrics
    geo_stats, time_stats = calculate_fairness_metrics(transaction_data)
    
    # Save results
    save_metrics(geo_stats, time_stats)
    generate_fairness_report(geo_stats, time_stats)
    
    print("Analysis complete. Output files generated:")
    print("- fairness_geo.csv (Geographic fairness metrics)")
    print("- fairness_time.csv (Temporal fairness metrics)")
    print("- fairness_report.png (Visual report)")