# Lab 13 - End-to-End MLOps Integration Report

## Objective

Integrate data versioning and validation, preprocessing, training, MLflow tracking and model registry, evaluation and quality gates, lifecycle decisions, deployment packaging, monitoring, conditional retraining triggers, failure injection, reproducibility, and engineering documentation into one traceable Titanic Survival MLOps platform.

## Integrated execution

`src/mlops_pipeline/12_monitor_trigger.py` (invoked directly or via `.github/workflows/lab13_monitoring.yml` on a self-hosted Windows runner) runs `12_drift_analysis.py`, checks for `data/raw/new_titanic.csv`, and invokes `src/mlops_pipeline/titanic_survival_pipeline.py` when drift and an incoming batch are present.

The main pipeline executes these 10 ordered stages:

1. Data Loading (`01_load_data.py`)
2. Raw Data Validation (`02_validate_data.py`)
3. Preprocessing (`03_preprocessing.py`)
4. Processed Output Validation (`04_validate_outputs.py`)
5. Training + MLflow + Model Registry (`05_train_registry.py`)
6. Model Evaluation (`06_model_evaluation.py`)
7. Model Quality Gate (`07_quality_gate.py`)
8. Automated Model Lifecycle (`08_automate_lifecycle.py`)
9. Deployment Update (`deploy_if_promoted.py`)
10. Production Inference (`09_inference.py`)

If the batch hash already exists in `data/raw/archive/dataset_lineage.json`, the pipeline stops safely after Stage 1 (`NO_NEW_DATA`) rather than retraining on duplicate data.

## Validation evidence

### New-batch ingestion and retraining

A simulated 30-row schema-valid batch (`data/raw/new_titanic.csv`, `PassengerId` `1001`-`1030`, SHA-256 `6aa10cc7db48191974c7be841adfb76af2b611267c553cf76143b92ece8c30d8`) passed Pandera schema validation and passenger ID non-overlap checks. `01_load_data.py` archived the original 891-row dataset as `data/raw/archive/titanic_v01.csv`, updated `data/raw/archive/dataset_lineage.json`, and appended the batch so `data/raw/titanic.csv` grew from 891 to 921 rows (742 training rows, 179 fixed held-out test rows).

### Duplicate-batch idempotency

A second run of `12_monitor_trigger.py` with the same `data/raw/new_titanic.csv` detected drift and matched the SHA-256 hash in `dataset_lineage.json`. `01_load_data.py` reported `already_ingested`, the orchestrator halted cleanly with `NO_NEW_DATA`, and `outputs/monitoring/monitor_trigger_report.json` recorded `status: "no_new_data_retraining_skipped"`, `new_data_ingested: false`, `retraining_executed: false`.

### Drift observations from the verified run

- Data drift: 4 of 7 monitored features (`Age`, `Fare`, `SibSp`, `Parch`) exceeded `KS >= 0.10`.
- Prediction distribution KS: `0.117318` (`DRIFT`).
- Probability distribution KS: `0.301676` (`DRIFT`).
- Concept drift: 4 of 5 monitored metrics degraded (`accuracy`, `precision`, `f1`, `roc_auc`).

### Candidate training and quality gate

Candidate model version 7 (`run_id`: `12c1c37f91484d1786e360d2a717786c`) was trained on 742 rows and evaluated on the 179-row fixed test split:

| Metric | Candidate v7 |
|---|---:|
| Accuracy | 0.8101 |
| Precision | 0.7536 |
| Recall | 0.7536 |
| F1 | 0.7536 |
| ROC-AUC | 0.8336 |

The recall quality gate (`>= 0.70`) passed (`0.7536 >= 0.7000`).

### Lifecycle and deployment decision

Production version 1 (`run_id`: `737141f08de64811a0d7882185647d10`) had recall `0.7681`, higher than candidate version 7's `0.7536`. The lifecycle manager retained Production v1 and archived v7 (`promoted: false`). `deploy_if_promoted.py` observed `promoted: false` and skipped deployment, after which `09_inference.py` generated Production predictions in `outputs/inference_predictions.csv`.

### Deployment bundle and cluster scope

`11_prepare_deployment.py` packaged Production version 1 into `deployment/model.pkl`, `deployment/preprocessor.pkl`, and `deployment/model_metadata.json`, verified locally via the 30 automated tests in `tests/test_api.py` and `tests/test_performance.py`. Live Docker image builds and Kubernetes cluster rollouts (`deployment/deploy_production.py`, `k8s/`) require Docker Desktop and `kubectl` (not installed on this host).

## Important artifacts

| Artifact | Purpose |
|---|---|
| `artifacts/data_load_report.json` | Ingestion outcome, SHA-256 hash, and row counts. |
| `data/raw/archive/dataset_lineage.json` | Batch hashes and dataset version history. |
| `outputs/drift/drift_analysis_report.csv` | Combined data, prediction, and concept drift summary. |
| `outputs/monitoring/monitor_trigger_report.json` | Monitoring and retraining trigger decision. |
| `reports/evaluation_report.json` | Candidate model evaluation metrics. |
| `reports/quality_gate_report.json` | Recall quality gate result. |
| `reports/lifecycle_report.json` | Candidate-versus-Production promotion decision. |
| `reports/failure_report.json` | Lab 9 failure injection and MLflow recovery evidence. |
| `reports/api_testing_report.json` | API functional and robustness test summary. |
| `reports/performance_report.json` | API latency, reliability, and throughput summary. |
| `deployment/model_metadata.json` | Identity and lineage of the packaged Production model. |
| `logs/titanic_survival_pipeline_run_log.json` | Stage-by-stage pipeline execution log. |
