import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


def train_and_track(run_name="RandomForest_Baseline", params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "class_weight": "balanced"
        }

    print(f"\n--- Starting MLflow Run: {run_name} ---")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    mlflow.set_experiment("Titanic_Survival_Prediction")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    with mlflow.start_run(run_name=run_name):

        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")

        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob)
        }

        mlflow.log_metrics(metrics)

        print(f"F1 = {metrics['f1_score']:.4f} | ROC-AUC = {metrics['roc_auc']:.4f}")

        # Confusion matrix
        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax_cm)
        ax_cm.set_title(f"Confusion Matrix - {run_name}")
        cm_path = f"artifacts/confusion_matrix_{run_name}.png"
        fig_cm.savefig(cm_path, bbox_inches="tight")
        plt.close(fig_cm)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        # ROC curve
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax_roc)
        ax_roc.set_title(f"ROC Curve - {run_name}")
        roc_path = f"artifacts/roc_curve_{run_name}.png"
        fig_roc.savefig(roc_path, bbox_inches="tight")
        plt.close(fig_roc)
        mlflow.log_artifact(roc_path, artifact_path="plots")

        metadata_path = "data/processed/dataset_metadata.json"
        if os.path.exists(metadata_path):
            mlflow.log_artifact(metadata_path, artifact_path="metadata")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        backup_name = "models/" + run_name.lower().replace(" ", "_") + ".pkl"
        joblib.dump(model, backup_name)

        print(f"Run '{run_name}' completed.")
        print(f"Local model: {backup_name}")

        return model, metrics


if __name__ == "__main__":
    train_and_track()
