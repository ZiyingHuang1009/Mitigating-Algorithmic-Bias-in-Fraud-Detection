# Overview
This repository contains the code, documentation, and resources for Jenny Huang's Major Research Project in Fairness-Aware Fraud Detection: Mitigating Algorithmic Bias in Transaction Systems. The project is currently in development, and this README will be updated as progress continues.

# Current Contents
For now, this repository includes:

main.py – Core functionality 

utils/ – Helper functions and utilities

data/ – Sample datasets or processing scripts 

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
