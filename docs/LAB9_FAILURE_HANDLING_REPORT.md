# Lab 9 - Pipeline Failure Handling and Recovery

## Goal

Intentionally break critical parts of the Titanic Survival MLOps pipeline and verify that the system fails fast on bad data or recovers through the MLflow Model Registry when local artifacts are missing.

## Run locally

```bash
python src/mlops_pipeline/10_failure_injection.py
```

## Scenarios

| Scenario | Expected behavior |
|---|---|
| Missing raw dataset (`data/raw/titanic.csv`) | HALT |
| Invalid datatype (`Fare = "INVALID_NUMBER"`) | HALT |
| Schema mismatch (missing `Sex` column) | HALT |
| Corrupted processed output (`NaN` in `X_train_final.npy`) | HALT |
| Missing local model artifact (`models/titanic_model.pkl`) | Detect failure, recover Production model from MLflow, then continue |

## Repeated execution

`config.yaml` controls the number of repetitions:

```yaml
lab9:
  repetitions: 3
```

The failure test creates:

```text
reports/failure_report.json
```

## Recovery path

```text
models/titanic_model.pkl missing
            |
            v
06_model_evaluation.py detects missing artifact
            |
            v
load MLflow Production model (models:/Titanic_Survival_Production_Model/Production)
            |
            v
restore local model artifact
            |
            v
run 06_model_evaluation.py again -> success
```

## Safety behavior

Corrupted input data is never silently repaired by the pipeline. Validation failures halt downstream stages so bad data cannot reach training.
