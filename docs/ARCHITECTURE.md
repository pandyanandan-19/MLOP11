# MLOps Titanic Survival Prediction - Architecture

## Purpose

This repository implements a Titanic Passenger Survival prediction workflow with strict schema validation, deterministic preprocessing and feature engineering, MLflow experiment and model registry tracking, a recall quality gate, automated lifecycle decisions, monitoring-triggered retraining, FastAPI model serving, and Docker/Kubernetes deployment configuration.

## End-to-end flow

```text
GitHub Actions (manual or scheduled)
              |
              v
Self-hosted Windows runner
              |
              v
12_monitor_trigger.py
   |          |
   |          +--> Read existing drift reports
   |               - data drift
   |               - prediction drift
   |               - concept/performance drift
   |
   +--> Check for data/raw/new_titanic.csv
              |
              v
titanic_survival_pipeline.py (when drift and a batch are present)
              |
              v
01 Data Loading + batch hash/identity checks
              |
              v
02 Raw validation --> 03 Preprocessing --> 04 Output validation
              |
              v
05 Training + MLflow/model registry
              |
              v
06 Evaluation --> 07 Recall quality gate (minimum 0.70)
              |
              v
08 Automated lifecycle decision
              |
        +-----+----------------+
        |                      |
   Not promoted             Promoted
        |                      |
        v                      v
Skip deployment        Prepare Production model bundle
                               |
                               v
                       Docker image + Kubernetes rollout
              |
              v
09 Production inference
```

## Main components

| Component | Responsibility |
|---|---|
| `src/mlops_pipeline/12_monitor_trigger.py` | Runs combined drift analysis, checks for an incoming batch, invokes the main pipeline when appropriate, and writes a monitoring-trigger report. |
| `src/mlops_pipeline/12_drift_analysis.py` | Combines data-, prediction-, and concept-drift reports into `outputs/drift/drift_analysis_report.csv`. |
| `src/mlops_pipeline/01_load_data.py` | Validates incoming schema, hashes the batch, prevents duplicate ingestion, rejects overlapping passenger IDs, archives the prior dataset, appends new rows, and records lineage. |
| `src/mlops_pipeline/titanic_survival_pipeline.py` | Runs ordered ML stages and stops safely if no new batch was ingested (unless `--force` is passed). |
| `src/mlops_pipeline/05_train_registry.py` | Trains a candidate Random Forest model, tracks the run in MLflow, attaches `preprocessor.pkl`, and registers a model version. |
| `src/mlops_pipeline/07_quality_gate.py` | Requires recall to meet the configured minimum of `0.70`. Passing this gate alone does not promote a model. |
| `src/mlops_pipeline/08_automate_lifecycle.py` | Compares the candidate with Production and records promotion/archive decisions in `reports/lifecycle_report.json`. |
| `src/mlops_pipeline/deploy_if_promoted.py` | Invokes deployment preparation and rollout only when the lifecycle report confirms promotion. |
| `src/mlops_pipeline/11_prepare_deployment.py` | Packages the registered Production model, its matching preprocessor, and metadata into `deployment/`. |
| `deployment/deploy_production.py` | Builds or reuses a versioned Docker image, updates Kubernetes, waits for rollout, and writes a deployment report. |
| `.github/workflows/lab13_monitoring.yml` | Starts the monitoring trigger through the self-hosted Windows runner. |

## Data ingestion and idempotency

`data/raw/new_titanic.csv` existing on disk does not prove that a batch is new. `01_load_data.py` computes a SHA-256 hash and compares it with `data/raw/archive/dataset_lineage.json`. If the hash already exists, it writes `already_ingested` and skips appending or retraining. For a genuinely new hash, it validates the schema and `PassengerId` uniqueness/non-overlap, archives the previous dataset snapshot, appends the batch to `data/raw/titanic.csv`, and updates lineage.

## Runtime dependencies and limits

- MLflow, data, and deployment artifacts live in the local project workspace unless an external `MLFLOW_TRACKING_URI` is configured.
- Drift analysis combines existing drift reports; upstream drift scripts refresh those CSV inputs.
- Docker Desktop and a Kubernetes cluster (`kubectl`) are required to execute container build and rollout commands in `deployment/deploy_production.py`.
- Deployment occurs only after lifecycle promotion. Passing training or the recall quality gate alone is insufficient.
