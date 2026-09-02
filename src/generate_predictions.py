import os
import joblib
import pandas as pd

from preprocessing import prepare_features

MODEL_PATH = "models/titanic_model.pkl"
TEST_PATH = "data/test.csv"
OUTPUT_PATH = "outputs/results/titanic_predictions.csv"


def main():
    os.makedirs("outputs/results", exist_ok=True)

    model = joblib.load(MODEL_PATH)

    test = pd.read_csv(TEST_PATH)
    passenger_ids = test["PassengerId"]

    X_test = prepare_features(test)

    predictions = model.predict(X_test).astype(int)

    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions
    })

    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Predictions saved to: {OUTPUT_PATH}")
    print("\nFirst 10 predictions:")
    print(submission.head(10))


if __name__ == "__main__":
    main()
