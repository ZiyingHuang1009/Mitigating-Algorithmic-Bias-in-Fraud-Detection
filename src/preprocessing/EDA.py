import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# 1. First define all helper functions
def configure_plots():
    # Configure plot style with robust settings
    # Set style with fallback parameters
    available_styles = plt.style.available
    if 'seaborn-v0_8' in available_styles:
        plt.style.use('seaborn-v0_8')
    elif 'seaborn' in available_styles:
        plt.style.use('seaborn')
    
    # Set grid and font properties
    plt.rcParams.update({
        'axes.grid': True,
        'axes.grid.axis': 'y',
        'grid.color': '0.85',
        'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.titlesize': 14,
        'figure.dpi': 300
    })
    
    # Set color palette
    sns.set_palette("husl")
    sns.set_style("whitegrid")

def load_data(path='data/raw/credit_card_fraud_dataset.csv'):
    # Load and validate dataset
    df = pd.read_csv(path)
    required_cols = ['TransactionDate', 'Amount', 'Location', 'IsFraud', 'TransactionType']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df

# 2. Then define the TemporalBinner class
class TemporalBinner(BaseEstimator, TransformerMixin):
    # Time binning with exact hour extraction
    def __init__(self, time_col='TransactionDate'):
        self.time_col = time_col
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        if self.time_col in X.columns:
            if X[self.time_col].str.contains('-').any():  # YYYY-MM-DD HH:MM
                dt = pd.to_datetime(X[self.time_col])
                X['Hour'] = dt.dt.hour + dt.dt.minute/60
            else:  # HH:MM.S format
                time_parts = X[self.time_col].str.split(':')
                X['Hour'] = time_parts.str[0].astype(int) + time_parts.str[1].astype(float)/60
            
            X['Time_Category'] = pd.cut(X['Hour'], bins=[0,6,12,18,24],
                                      labels=['Night','Morning','Afternoon','Evening'],
                                      right=False)
        return X

# 3. Then define visualization functions
def save_exact_values(df, filename='data/reports/exact_values.txt'):
    # Save exact numerical values for all visualizations
    with open(filename, 'w') as f:
        # Class distribution
        class_counts = df['IsFraud'].value_counts()
        f.write("=== CLASS DISTRIBUTION ===\n")
        f.write(f"Legitimate: {class_counts[0]} ({class_counts[0]/len(df):.4%})\n")
        f.write(f"Fraud: {class_counts[1]} ({class_counts[1]/len(df):.4%})\n\n")
        
        # Transaction type rates
        if 'TransactionType' in df.columns:
            type_rates = df.groupby('TransactionType')['IsFraud'].mean()
            f.write("=== TRANSACTION TYPE RATES ===\n")
            for t, rate in type_rates.items():
                f.write(f"{t}: {rate:.6f} ({rate*100:.4f}%)\n")
            f.write("\n")
        
        # Hourly rates
        if 'Hour' in df.columns:
            hourly = df.groupby(np.floor(df['Hour']))['IsFraud'].mean()
            f.write("=== HOURLY RATES ===\n")
            for hour, rate in hourly.items():
                f.write(f"{int(hour)}:00: {rate:.6f}\n")

def generate_figure1(df):
    # Enhanced 4-panel visualization
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Figure 1: Exploratory Data Analysis", y=1.02)
    
    # Panel 1A: Amount Distribution
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    sns.kdeplot(data=df, x='Amount', hue='IsFraud', common_norm=False, 
               fill=True, alpha=0.5, ax=ax1)
    ax1.set_title('A) Transaction Amount Distribution')
    ax1.set_xlabel('Amount ($)')
    ax1.set_ylabel('Density')
    
    # Panel 1B: Class Imbalance
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    counts = df['IsFraud'].value_counts()
    ax2.pie(counts, labels=['Legitimate', 'Fraud'], colors=['#66c2a5', '#fc8d62'],
           autopct=lambda p: f'{p:.2f}%\n({p*sum(counts)/100:.0f} txns)')
    ax2.set_title('B) Class Distribution')
    
    # Panel 1C: Enhanced Transaction Type View
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    if 'TransactionType' in df.columns:
        type_data = df.groupby('TransactionType')['IsFraud'].agg(['mean', 'count'])
        bars = ax3.bar(type_data.index, type_data['mean']*100, color=['#1f77b4', '#ff7f0e'])
        ax3.set_title('C) Fraud Rate by Transaction Type')
        ax3.set_ylabel('Fraud Rate (%)')
        ax3.set_ylim(0, max(0.02, type_data['mean'].max() * 150))
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height,
                    f'{height:.4f}%', ha='center', va='bottom', fontsize=9)
    
    # Panel 1D: Geographic Analysis
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    if 'Location' in df.columns:
        loc_rates = df.groupby('Location')['IsFraud'].mean().sort_values() * 100
        loc_rates.plot.bar(ax=ax4, color='#9467bd')
        ax4.set_title('D) Fraud Rate by Location')
        ax4.set_ylabel('Fraud Rate (%)')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for i, v in enumerate(loc_rates):
            ax4.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    fig.savefig('data/reports/figure1_enhanced.png', bbox_inches='tight')
    plt.close()

def generate_figure2(df):
    # Temporal analysis with exact values
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("Figure 2: Temporal Analysis", y=1.05)
    
    # Panel 2A: Hourly Fluctuations
    ax1 = plt.subplot(1, 2, 1)
    if 'Hour' in df.columns:
        hourly = df.groupby(np.floor(df['Hour']))['IsFraud'].mean() * 100
        line = hourly.plot(ax=ax1, marker='o', color='#e377c2')
        ax1.set_title('A) Hourly Fraud Probability')
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Fraud Rate (%)')
        ax1.grid(True)
        
        # Annotate values
        for i, v in enumerate(hourly):
            ax1.text(i, v + 0.001, f"{v:.4f}", ha='center', va='bottom', fontsize=8)
    
    # Panel 2B: Time Category Comparison
    ax2 = plt.subplot(1, 2, 2)
    if 'Time_Category' in df.columns:
        sns.barplot(
            data=df, 
            x='Time_Category', 
            y='IsFraud', 
            hue='Time_Category',
            estimator=np.mean, 
            errorbar=('ci', 95), 
            ax=ax2,
            order=['Night','Morning','Afternoon','Evening'],
            palette='rocket', 
            legend=False
        )
        ax2.set_title('B) Fraud Rate by Time Category')
        ax2.set_xlabel('')
        ax2.set_ylabel('Fraud Probability')
        
        # Add value labels
        for p in ax2.patches:
            ax2.text(p.get_x() + p.get_width()/2, p.get_height(),
                    f'{p.get_height():.6f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig.savefig('data/reports/figure2_enhanced.png', bbox_inches='tight')
    plt.close()

# 4. Finally define the main analysis function
def run_analysis():
    # Main analysis pipeline
    import warnings
    warnings.filterwarnings("ignore")
    
    configure_plots()
    df = load_data()
    print(f"Analyzing {len(df)} transactions ({df['IsFraud'].mean():.4%} fraud)")
    
    # Process data
    df = TemporalBinner().transform(df)
    
    # Generate outputs
    save_exact_values(df)
    generate_figure1(df)
    generate_figure2(df)
    
    print("Analysis completed. Output files:")
    print("- data/reports/exact_values.txt")
    print("- data/reports/figure1_enhanced.png")
    print("- data/reports/figure2_enhanced.png")

# 5. Entry point at the very end
if __name__ == '__main__':
    run_analysis()
