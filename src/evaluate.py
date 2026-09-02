import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    RocCurveDisplay
)

from preprocessing import load_data


os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)


def main():
    X, y, _ = load_data()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model_path = "models/titanic_model.pkl"
    model = joblib.load(model_path)

    y_pred = model.predict(X_valid)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_valid)[:, 1]
    else:
        y_prob = None

    metrics = {
        "Accuracy": accuracy_score(y_valid, y_pred),
        "Precision": precision_score(y_valid, y_pred, zero_division=0),
        "Recall": recall_score(y_valid, y_pred, zero_division=0),
        "F1 Score": f1_score(y_valid, y_pred, zero_division=0)
    }

    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_valid, y_prob)

    print("=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_valid, y_pred, zero_division=0))

    pd.DataFrame([metrics]).to_csv(
        "outputs/results/final_model_metrics.csv",
        index=False
    )

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_valid,
        y_pred
    )
    plt.title("Confusion Matrix - Titanic Survival")
    plt.tight_layout()
    plt.savefig("outputs/plots/confusion_matrix.png", dpi=200)
    plt.close()

    # ROC curve
    if y_prob is not None:
        RocCurveDisplay.from_predictions(
            y_valid,
            y_prob
        )
        plt.title("ROC Curve - Titanic Survival")
        plt.tight_layout()
        plt.savefig("outputs/plots/roc_curve.png", dpi=200)
        plt.close()

    print("\nEvaluation files saved in outputs/")


if __name__ == "__main__":
    main()
