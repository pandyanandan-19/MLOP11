import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.preprocess import run_preprocessing
from train_mlflow import train_and_track


if __name__ == "__main__":
    print("=" * 60)
    print("LAB 4 - MLflow TRACKING PIPELINE")
    print("=" * 60)

    run_preprocessing()

    train_and_track(
        run_name="RandomForest_Baseline",
        params={
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }
    )

    train_and_track(
        run_name="RandomForest_Shallow",
        params={
            "n_estimators": 200,
            "max_depth": 5,
            "random_state": 42,
            "class_weight": "balanced"
        }
    )

    print("\nMLflow tracking pipeline completed.")
