import sys
import os
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now use absolute imports
from src.core.pipeline import Pipeline
from src.utils import setup_logger

def parse_args():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument(
        "--phase", 
        type=str, 
        required=True,
        choices=["train", "evaluation", "fairness"],
        help="Pipeline phase to execute"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="smote",
        choices=["smote", "adasyn", "ros"],
        help="Data version to use"
    )
    return parser.parse_args()

def main():
    logger = setup_logger(__name__)
    args = parse_args()
    
    pipeline = Pipeline(version=args.version)
    
    # Map 'train' to 'evaluation' since that's what your pipeline expects
    phase = "evaluation" if args.phase == "train" else args.phase
    results = pipeline.run(phase)
    
    logger.info(f"Completed {phase} phase with {args.version} version")
    logger.info(f"Results saved to: {results[1]}")

if __name__ == "__main__":
    main()
