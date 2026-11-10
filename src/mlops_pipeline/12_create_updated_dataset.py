from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FEATURES_PATH = Path("data/processed/X_train_raw.csv")
TRAIN_LABELS_PATH = Path("data/processed/y_train.npy")
TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")

NEW_BATCH_PATH = Path("data/monitoring/drift/post_drift_training_data.csv")
UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
SUMMARY_PATH = Path("outputs/retraining/updated_dataset_summary.csv")

RANDOM_SEED = 42
NEW_BATCH_SIZE = 179


def load_original_training_data():
    X_train = pd.read_csv(TRAIN_FEATURES_PATH)
    y_train = np.load(TRAIN_LABELS_PATH).astype(int)
    X_test = pd.read_csv(TEST_FEATURES_PATH)

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Training feature/label mismatch: X={len(X_train)}, y={len(y_train)}"
        )

    return X_train, y_train, X_test


def create_post_drift_batch(X_train, y_train):
    rng = np.random.default_rng(RANDOM_SEED)

    sample_indices = rng.choice(len(X_train), size=NEW_BATCH_SIZE, replace=True)
    X_new = X_train.iloc[sample_indices].copy().reset_index(drop=True)
    y_new = y_train[sample_indices].copy()

    if "Age" in X_new.columns:
        X_new["Age"] = (
            X_new["Age"].fillna(X_new["Age"].median()) * rng.normal(1.25, 0.12, len(X_new)) + 4.0
        ).clip(0.42, 80.0).round(2)

    if "Fare" in X_new.columns:
        X_new["Fare"] = (
            X_new["Fare"].fillna(X_new["Fare"].median()) * rng.normal(1.40, 0.15, len(X_new)) + 5.0
        ).clip(0.0).round(4)

    if "Pclass" in X_new.columns:
        X_new["Pclass"] = rng.choice([1, 2, 3], size=len(X_new), p=[0.50, 0.25, 0.25])

    if "Sex" in X_new.columns:
        X_new["Sex"] = rng.choice(["male", "female"], size=len(X_new), p=[0.40, 0.60])

    if "Embarked" in X_new.columns:
        X_new["Embarked"] = rng.choice(["S", "C", "Q"], size=len(X_new), p=[0.35, 0.45, 0.20])

    # Simulate changed relationship in post-drift environment
    if "Pclass" in X_new.columns:
        pclass3 = X_new["Pclass"].eq(3)
        candidate_indices = np.where(pclass3 & (y_new == 1))[0]
        flip_count = int(len(candidate_indices) * 0.40)
        if flip_count > 0:
            selected = rng.choice(candidate_indices, size=flip_count, replace=False)
            y_new[selected] = 0

    if "Fare" in X_new.columns:
        high_fare = X_new["Fare"] >= X_new["Fare"].quantile(0.70)
        candidate_indices = np.where(high_fare & (y_new == 0))[0]
        flip_count = int(len(candidate_indices) * 0.35)
        if flip_count > 0:
            selected = rng.choice(candidate_indices, size=flip_count, replace=False)
            y_new[selected] = 1

    return X_new, y_new


def validate_new_batch(X_new, y_new, X_test):
    expected_columns = list(X_test.columns)

    if list(X_new.columns) != expected_columns:
        raise ValueError("Feature schema mismatch between new batch and test feature schema.")

    if len(X_new) != len(y_new):
        raise ValueError("New batch feature/label count mismatch.")

    if X_new.isnull().all().any():
        raise ValueError("One or more columns contain only null values.")

    if "Age" in X_new.columns and ((X_new["Age"].dropna() < 0) | (X_new["Age"].dropna() > 100)).any():
        raise ValueError("Age contains values outside [0, 100].")

    if "Fare" in X_new.columns and (X_new["Fare"].dropna() < 0).any():
        raise ValueError("Fare contains negative values.")

    if "Pclass" in X_new.columns and not X_new["Pclass"].isin([1, 2, 3]).all():
        raise ValueError("Pclass contains values outside {1, 2, 3}.")

    test_hashes = set(pd.util.hash_pandas_object(X_test, index=False).astype("uint64"))
    new_hashes = pd.util.hash_pandas_object(X_new, index=False).astype("uint64")
    exact_overlap = sum(value in test_hashes for value in new_hashes)

    if exact_overlap > 0:
        raise ValueError(f"Detected {exact_overlap} exact feature-row overlaps with the test set.")

    return exact_overlap


def create_updated_dataset(X_train, y_train, X_new, y_new):
    original_train = X_train.copy()
    original_train["Survived"] = y_train

    new_batch = X_new.copy()
    new_batch["Survived"] = y_new

    updated = pd.concat([original_train, new_batch], ignore_index=True)
    return updated, original_train, new_batch


def save_summary(original_train, new_batch, updated):
    rows = [
        {
            "dataset": "original_training",
            "rows": len(original_train),
            "survived": int((original_train["Survived"] == 1).sum()),
            "not_survived": int((original_train["Survived"] == 0).sum()),
            "survival_rate": round(float(original_train["Survived"].mean()), 6),
        },
        {
            "dataset": "post_drift_batch",
            "rows": len(new_batch),
            "survived": int((new_batch["Survived"] == 1).sum()),
            "not_survived": int((new_batch["Survived"] == 0).sum()),
            "survival_rate": round(float(new_batch["Survived"].mean()), 6),
        },
        {
            "dataset": "updated_training",
            "rows": len(updated),
            "survived": int((updated["Survived"] == 1).sum()),
            "not_survived": int((updated["Survived"] == 0).sum()),
            "survival_rate": round(float(updated["Survived"].mean()), 6),
        },
    ]

    summary = pd.DataFrame(rows)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def main():
    print("=" * 70)
    print("UPDATED TRAINING DATASET CREATION")
    print("=" * 70)

    X_train, y_train, X_test = load_original_training_data()

    print("\nORIGINAL DATA")
    print("-" * 70)
    print(f"Training features : {X_train.shape}")
    print(f"Training labels   : {y_train.shape}")
    print(f"Test features     : {X_test.shape}")

    X_new, y_new = create_post_drift_batch(X_train, y_train)
    exact_overlap = validate_new_batch(X_new, y_new, X_test)

    print("\nPOST-DRIFT BATCH")
    print("-" * 70)
    print(f"Rows              : {len(X_new)}")
    print(f"Columns           : {len(X_new.columns)}")
    print(f"Survived          : {(y_new == 1).sum()}")
    print(f"Did not survive   : {(y_new == 0).sum()}")
    print(f"Test-set overlap  : {exact_overlap}")

    updated, original_train, new_batch = create_updated_dataset(X_train, y_train, X_new, y_new)

    NEW_BATCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    UPDATED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    new_batch.to_csv(NEW_BATCH_PATH, index=False)
    updated.to_csv(UPDATED_DATASET_PATH, index=False)
    summary = save_summary(original_train, new_batch, updated)

    print("\nUPDATED TRAINING DATASET")
    print("-" * 70)
    print(f"Rows              : {len(updated)}")
    print(f"Columns           : {len(updated.columns)}\n")
    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("FILES CREATED")
    print("=" * 70)
    print(f"Post-drift batch  : {NEW_BATCH_PATH}")
    print(f"Updated dataset   : {UPDATED_DATASET_PATH}")
    print(f"Summary           : {SUMMARY_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
