# Overview
This repository contains the code, documentation, and resources for Jenny Huang's Major Research Project in Fairness-Aware Fraud Detection: Mitigating Algorithmic Bias in Transaction Systems. A machine learning pipeline for credit card fraud detection that incorporates fairness-aware techniques to mitigate potential biases in the model predictions. The project is currently in development, and this README will be updated as progress continues.

# Key Features
- **Advanced Fraud Detection**: XGBoost, Balanced Random Forest, and Isolation Forest models
- 
- **Fairness Mitigation**: Demographic parity and equalized odds difference metrics
- 
- **Comprehensive Evaluation**: Fraud-specific metrics with threshold optimization
- 
- **Modular Pipeline**: Clean separation of data loading, feature engineering, training, and evaluation
- 
- **Bias Analysis**: Protected attribute analysis (time, location)


# Process
# Phase 1: Data Preparation
python -m src.preprocessing.EDA

python -m src.preprocessing.preprocessor

# Phase 2: Main Pipeline
python -m src.core.main --phase fairness --version smote

python -m src.core.main --phase fairness --version adasyn

python -m src.core.main --phase fairness --version ros

# Phase 3:  Model Evaluation
python -m src.core.main --phase evaluation --version smote

python -m src.core.main --phase evaluation --version adasyn

python -m src.core.main --phase evaluation --version ros

# Phase 4: Update Evaluation 
python -m src.core.main --phase train --version smote

python -m src.core.main --phase train --version adasyn

python -m src.core.main --phase train --version ros
