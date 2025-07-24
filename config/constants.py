# EU AI Act Compliance Thresholds
SPD_THRESHOLD = 0.1
EOD_THRESHOLD = 0.05

# Protected Attributes
PROTECTED_ATTRS = {
    'time': ['TimeCategory_Morning', 'TimeCategory_Night'],
    'location': ['Location_Philadelphia', 'Location_NYC']
}

# Privileged/Unprivileged Groups
PRIVILEGED_GROUPS = [{'TimeCategory_Morning': 1}]
UNPRIVILEGED_GROUPS = [{'TimeCategory_Night': 1}]
RESAMPLING_METHODS = ['ADASYN', 'SMOTE', 'ROS', 'KDE-SMOTE']