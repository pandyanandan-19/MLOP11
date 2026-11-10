import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.validate_data import validate_schema


def simulate_corrupted_data(df):
    """Simulates real-world data corruption scenarios across multiple Titanic columns."""
    print("[INFO] Injecting simulated faults into Titanic data...")
    corrupted_df = df.copy()

    # 1. Invalid numerical values (negative Age)
    corrupted_df.loc[0, "Age"] = -5.0

    # 2. Invalid categorical value (Embarked)
    corrupted_df.loc[1, "Embarked"] = "X"

    # 3. Invalid target label (Survived not in {0, 1})
    corrupted_df.loc[2, "Survived"] = 99

    # 4. Datatype mismatch (injecting string into Fare float column)
    corrupted_df["Fare"] = corrupted_df["Fare"].astype(object)
    corrupted_df.loc[3, "Fare"] = "One Hundred"

    # 5. Schema inconsistency (unexpected extra column)
    corrupted_df["unexpected_column"] = 123

    # 6. Missing required attribute (dropping Sex column)
    corrupted_df = corrupted_df.drop(columns=["Sex"])

    return corrupted_df


if __name__ == "__main__":
    data_path = "data/raw/titanic.csv"

    try:
        raw_df = pd.read_csv(data_path)
        corrupted_df = simulate_corrupted_data(raw_df)

        print("\n--- Running Validation on Corrupted Data ---")
        validate_schema(corrupted_df, output_report_name="corrupted_simulation_errors.csv")
    except FileNotFoundError:
        print(f"[ERROR] Could not find data file at {data_path}.")
