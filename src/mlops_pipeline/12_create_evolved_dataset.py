from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_PATH = Path("data/processed/X_test_raw.csv")
OUTPUT_PATH = Path("data/monitoring/drift/evolved_data.csv")

RANDOM_SEED = 42


def create_evolved_dataset():
    rng = np.random.default_rng(RANDOM_SEED)
    df = pd.read_csv(SOURCE_PATH).copy()

    # 1. Numerical drift: Age (older demographic shift)
    if "Age" in df.columns:
        df["Age"] = (
            df["Age"].fillna(df["Age"].median()) * rng.normal(1.25, 0.12, len(df)) + 4.0
        ).clip(lower=0.42, upper=80.0).round(2)

    # 2. Numerical drift: Fare (higher ticket pricing shift)
    if "Fare" in df.columns:
        df["Fare"] = (
            df["Fare"].fillna(df["Fare"].median()) * rng.normal(1.40, 0.15, len(df)) + 5.0
        ).clip(lower=0.0).round(4)

    # 3. Numerical drift: SibSp & Parch
    if "SibSp" in df.columns:
        df["SibSp"] = rng.choice([0, 1, 2, 3], size=len(df), p=[0.50, 0.30, 0.15, 0.05])
    if "Parch" in df.columns:
        df["Parch"] = rng.choice([0, 1, 2], size=len(df), p=[0.60, 0.25, 0.15])

    # 4. Categorical drift: Pclass
    if "Pclass" in df.columns:
        df["Pclass"] = rng.choice([1, 2, 3], size=len(df), p=[0.50, 0.25, 0.25])

    # 5. Categorical drift: Sex
    if "Sex" in df.columns:
        df["Sex"] = rng.choice(["male", "female"], size=len(df), p=[0.40, 0.60])

    # 6. Categorical drift: Embarked
    if "Embarked" in df.columns:
        df["Embarked"] = rng.choice(["S", "C", "Q"], size=len(df), p=[0.35, 0.45, 0.20])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("EVOLVED DATASET CREATED")
    print("=" * 60)
    print(f"\nSource      : {SOURCE_PATH}")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Rows        : {len(df)}")
    print(f"Columns     : {len(df.columns)}")
    print("\nModified features:")
    print("- Age")
    print("- Fare")
    print("- SibSp")
    print("- Parch")
    print("- Pclass")
    print("- Sex")
    print("- Embarked")


if __name__ == "__main__":
    create_evolved_dataset()
