from pathlib import Path

import numpy as np
import pandas as pd

FEATURES_PATH = Path("data/monitoring/drift/evolved_data.csv")
LABELS_PATH = Path("data/processed/y_test.npy")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_labeled_data.csv")

RANDOM_SEED = 42


def create_concept_drift_dataset():
    rng = np.random.default_rng(RANDOM_SEED)

    X = pd.read_csv(FEATURES_PATH)
    y = np.load(LABELS_PATH).astype(int)

    if len(X) != len(y):
        raise ValueError(f"Feature/label mismatch: X={len(X)}, y={len(y)}")

    y_evolved = y.copy()

    # Simulate concept drift: relationship between passenger attributes and survival shifts.
    # 1. Third-class passengers experience lower survival outcomes.
    if "Pclass" in X.columns:
        pclass3_indices = np.where((X["Pclass"] == 3) & (y_evolved == 1))[0]
        flip_count = int(len(pclass3_indices) * 0.50)
        if flip_count > 0:
            selected = rng.choice(pclass3_indices, size=flip_count, replace=False)
            y_evolved[selected] = 0

    # 2. High-fare passengers become more likely to survive.
    if "Fare" in X.columns:
        high_fare_indices = np.where(
            (X["Fare"] > X["Fare"].quantile(0.70)) & (y_evolved == 0)
        )[0]
        flip_count = int(len(high_fare_indices) * 0.45)
        if flip_count > 0:
            selected = rng.choice(high_fare_indices, size=flip_count, replace=False)
            y_evolved[selected] = 1

    result = X.copy()
    result["Survived"] = y_evolved

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("=" * 70)
    print("CONCEPT DRIFT DATASET CREATED")
    print("=" * 70)
    print(f"Feature source : {FEATURES_PATH}")
    print(f"Label source   : {LABELS_PATH}")
    print(f"Output         : {OUTPUT_PATH}")
    print(f"Rows           : {len(result)}")
    print(f"Columns        : {len(result.columns)}")

    print("\nBaseline label distribution:")
    print(pd.Series(y).value_counts().sort_index())

    print("\nEvolved label distribution:")
    print(pd.Series(y_evolved).value_counts().sort_index())

    print("\nLabel changes:")
    print(f"Changed labels : {(y != y_evolved).sum()}")


if __name__ == "__main__":
    create_concept_drift_dataset()
