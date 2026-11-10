import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocess import run_preprocessing
from src.train import train_model
from src.evaluate import run_evaluation


if __name__ == "__main__":
    print("=" * 60)
    print("LAB 3 - BASELINE TITANIC SURVIVAL PREDICTION PIPELINE")
    print("=" * 60)

    run_preprocessing()
    train_model()
    run_evaluation()

    print("\nLab 3 pipeline completed successfully.")
