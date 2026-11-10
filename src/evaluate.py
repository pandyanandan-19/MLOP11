import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run_evaluation():
    print("Starting Model Evaluation...")

    # 1. Load test data and the trained model
    X_test_final = np.load('data/processed/X_test_final.npy')
    y_test = np.load('data/processed/y_test.npy')
    model = joblib.load('models/random_forest_baseline.pkl')

    # 2. Make predictions
    y_pred = model.predict(X_test_final)

    # 3. Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n--- Model Evaluation Report ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print("-------------------------------\n")

    # 4. Error Analysis Export
    data_path = 'data/raw/titanic.csv'
    if not os.path.exists(data_path):
        data_path = 'data/train.csv'
    df = pd.read_csv(data_path)

    _, X_test_raw = train_test_split(
        df.drop('Survived', axis=1), test_size=0.2, random_state=42, stratify=df['Survived']
    )

    errors_df = X_test_raw.copy()
    errors_df['Actual_Survived'] = y_test
    errors_df['Predicted_Survived'] = y_pred

    false_negatives = errors_df[(errors_df['Actual_Survived'] == 1) & (errors_df['Predicted_Survived'] == 0)]
    false_positives = errors_df[(errors_df['Actual_Survived'] == 0) & (errors_df['Predicted_Survived'] == 1)]

    os.makedirs('outputs', exist_ok=True)
    false_negatives.to_csv('outputs/false_negatives.csv', index=False)
    false_positives.to_csv('outputs/false_positives.csv', index=False)

    print("Evaluation complete and error analysis files saved!")
    print(f"Saved: outputs/false_negatives.csv ({len(false_negatives)} rows)")
    print(f"Saved: outputs/false_positives.csv ({len(false_positives)} rows)")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1}


main = run_evaluation


if __name__ == "__main__":
    run_evaluation()
