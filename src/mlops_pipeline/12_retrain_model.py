import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from api.preprocessing import clean_prediction_data

UPDATED_DATASET_PATH = Path("data/processed/updated_training_data.csv")
TEST_FEATURES_PATH = Path("data/processed/X_test_raw.csv")
TEST_LABELS_PATH = Path("data/processed/y_test.npy")

MODEL_V2_PATH = Path("models/retrained_model_v2.pkl")
PREPROCESSOR_V2_PATH = Path("models/retrained_preprocessor_v2.pkl")
REPORT_PATH = Path("outputs/retraining/retraining_report.json")

NUM_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked", "Title", "IsAlone"]


def configure_mlflow(config):
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or config["mlflow"].get("tracking_uri")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)


def get_current_production_version(model_name):
    client = MlflowClient()
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prod = [v for v in versions if v.current_stage == "Production"]
        return int(prod[0].version) if prod else None
    except Exception:
        return None


def build_preprocessor(X_train_clean):
    num_cols = [c for c in NUM_FEATURES if c in X_train_clean.columns]
    cat_cols = [c for c in CAT_FEATURES if c in X_train_clean.columns]

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ]
    )


def evaluate_model(model, preprocessor, X_test, y_test):
    X_test_clean = clean_prediction_data(X_test)
    X_test_transformed = preprocessor.transform(X_test_clean)

    predictions = model.predict(X_test_transformed)
    probabilities = model.predict_proba(X_test_transformed)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }


def main():
    print("=" * 70)
    print("LAB 12 - MODEL RETRAINING")
    print("=" * 70)

    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    configure_mlflow(config)

    if not UPDATED_DATASET_PATH.exists():
        raise FileNotFoundError(f"Updated training dataset not found: {UPDATED_DATASET_PATH}")
    if not TEST_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Test features not found: {TEST_FEATURES_PATH}")
    if not TEST_LABELS_PATH.exists():
        raise FileNotFoundError(f"Test labels not found: {TEST_LABELS_PATH}")

    updated = pd.read_csv(UPDATED_DATASET_PATH)
    X_test = pd.read_csv(TEST_FEATURES_PATH)
    y_test = np.load(TEST_LABELS_PATH).astype(int)

    print("\nDATASET INFORMATION")
    print("-" * 70)
    print(f"Updated training data : {updated.shape}")
    print(f"Held-out test data    : {X_test.shape}")
    print(f"Test labels           : {y_test.shape}")

    if "Survived" not in updated.columns:
        raise ValueError("Updated training dataset must contain Survived column.")

    X_train = updated.drop(columns=["Survived"])
    y_train = (
        updated["Survived"]
        .apply(lambda x: 1 if str(x).strip().lower() in {"1", "yes", "true"} else 0)
        .astype(int)
        .to_numpy()
    )

    if len(X_train) != len(y_train):
        raise ValueError("Training feature/label count mismatch.")
    if len(X_test) != len(y_test):
        raise ValueError("Test feature/label count mismatch.")

    model_cfg = config["model"]
    params = {
        "n_estimators": model_cfg["n_estimators"],
        "max_depth": model_cfg["max_depth"],
        "random_state": model_cfg["random_state"],
        "class_weight": model_cfg["class_weight"],
    }

    print("\nMODEL PARAMETERS")
    print("-" * 70)
    for name, value in params.items():
        print(f"{name:15}: {value}")

    print("\nFITTING V2 PREPROCESSOR")
    print("-" * 70)

    X_train_clean = clean_prediction_data(X_train)
    preprocessor = build_preprocessor(X_train_clean)
    X_train_transformed = preprocessor.fit_transform(X_train_clean)

    print(f"Original features       : {X_train.shape[1]}")
    print(f"Transformed features    : {X_train_transformed.shape[1]}")

    print("\nTRAINING MODEL V2")
    print("-" * 70)
    model = RandomForestClassifier(**params)
    model.fit(X_train_transformed, y_train)
    print("Training completed.")

    print("\nEVALUATING MODEL V2 ON ORIGINAL HELD-OUT TEST SET")
    print("-" * 70)
    metrics = evaluate_model(model, preprocessor, X_test, y_test)
    for name, value in metrics.items():
        print(f"{name.upper():10}: {value:.4f}")

    MODEL_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_V2_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_V2_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_V2_PATH)

    print("\nARTIFACTS SAVED")
    print("-" * 70)
    print(f"Model V2        : {MODEL_V2_PATH}")
    print(f"Preprocessor V2 : {PREPROCESSOR_V2_PATH}")

    prev_prod_version = get_current_production_version(config["registry"]["model_name"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="Lab12_Retraining_Model_V2") as run:
        mlflow.log_params(params)
        mlflow.log_param("model_family", "RandomForest")
        mlflow.log_param("lab", "Lab 12 - Continuous Training")
        mlflow.log_param("training_dataset", str(UPDATED_DATASET_PATH))
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))
        mlflow.log_param("training_survival_rate", float(y_train.mean()))
        mlflow.log_param("production_model_version_before_retraining", str(prev_prod_version))

        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(UPDATED_DATASET_PATH), artifact_path="retraining_dataset")
        mlflow.log_artifact(str(MODEL_V2_PATH), artifact_path="retrained_model")
        mlflow.log_artifact(str(PREPROCESSOR_V2_PATH), artifact_path="preprocessing_pipeline_v2")

        print("\nMLFLOW RETRAINING RUN")
        print("-" * 70)
        print(f"Experiment : {config['mlflow']['experiment_name']}")
        print(f"Run ID     : {run.info.run_id}")

    report = {
        "stage": "Lab 12 - Retraining",
        "model_version": "V2",
        "previous_production_version": prev_prod_version,
        "training_dataset": str(UPDATED_DATASET_PATH),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "training_survival_rate": round(float(y_train.mean()), 6),
        "model_parameters": params,
        "metrics": {key: round(float(value), 6) for key, value in metrics.items()},
        "model_path": str(MODEL_V2_PATH),
        "preprocessor_path": str(PREPROCESSOR_V2_PATH),
        "mlflow_run_id": run.info.run_id,
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print("\n" + "=" * 70)
    print("RETRAINING COMPLETE")
    print("=" * 70)
    print(f"Report : {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
