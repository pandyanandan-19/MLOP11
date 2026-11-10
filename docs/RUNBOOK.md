# Runbook - Titanic Survival MLOps

## Before running

1. Ensure you are in the repository root and using the `.venv` Python environment:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. For Docker/Kubernetes operations, start Docker Desktop and wait until Docker and Kubernetes are ready:
   ```powershell
   docker info
   kubectl get nodes
   kubectl get deployment titanic-api
   ```
3. For GitHub Actions self-hosted monitoring runs, start the runner in an interactive PowerShell window:
   ```powershell
   cd C:\actions-runner
   .\run.cmd
   ```

## Run the main MLOps pipeline

Run all 10 stages (including forced retraining when no new batch is present):
```powershell
python src/mlops_pipeline/titanic_survival_pipeline.py --force
```

Run in batch-gated mode (stops cleanly if `data/raw/new_titanic.csv` is absent or already ingested):
```powershell
python src/mlops_pipeline/titanic_survival_pipeline.py
```

## Run monitoring and retraining trigger

1. Generate/refresh drift datasets and reports:
   ```powershell
   python src/mlops_pipeline/12_create_evolved_dataset.py
   python src/mlops_pipeline/12_data_drift.py
   python src/mlops_pipeline/12_prediction_drift.py
   python src/mlops_pipeline/12_create_concept_drift_dataset.py
   python src/mlops_pipeline/12_concept_drift.py
   ```
2. Run combined drift analysis and conditional trigger:
   ```powershell
   python src/mlops_pipeline/12_monitor_trigger.py
   ```

## Interpret the monitoring report

Inspect `outputs/monitoring/monitor_trigger_report.json`:

| Status | Meaning |
|---|---|
| `no_drift` | Combined drift analysis did not detect drift; pipeline is not called. |
| `drift_waiting_for_data` | Drift was detected, but `data/raw/new_titanic.csv` is missing. |
| `no_new_data_retraining_skipped` | `new_titanic.csv` exists, but its SHA-256 hash was already ingested. |
| `retraining_completed` | New batch was ingested and the pipeline completed. Check `reports/lifecycle_report.json` for promotion status. |
| `drift_analysis_failed` | Drift analysis failed; inspect upstream drift reports in `outputs/drift/`. |

## Important reports

- `artifacts/data_load_report.json` - ingestion status and batch SHA-256 hash.
- `data/raw/archive/dataset_lineage.json` - ingested batches and archived dataset versions.
- `outputs/drift/drift_analysis_report.csv` - combined drift summary.
- `outputs/monitoring/monitor_trigger_report.json` - monitoring trigger decision.
- `reports/evaluation_report.json` - candidate evaluation metrics.
- `reports/quality_gate_report.json` - recall quality gate check.
- `reports/lifecycle_report.json` - candidate-vs-Production promotion decision.
- `reports/failure_report.json` - Lab 9 failure injection & recovery summary.
- `reports/deployment_automation_report.json` - deployment bundle / rollout outcome.
- `logs/titanic_survival_pipeline_run_log.json` - stage-by-stage pipeline log.

## Troubleshooting

### Retraining is skipped
Check `artifacts/data_load_report.json` and `data/raw/archive/dataset_lineage.json`. `already_ingested` means the exact SHA-256 hash of `data/raw/new_titanic.csv` was already consumed. Supply a new schema-valid batch with non-overlapping `PassengerId` values to trigger ingestion.

### Model was not deployed
Inspect `reports/lifecycle_report.json`. If `promoted` is `false`, `deploy_if_promoted.py` intentionally skips deployment because the candidate did not outperform the current Production model on the decision metric.

### UnicodeEncodeError on Windows paths
Subprocesses run with UTF-8 stdout reconfiguration; if invoking scripts manually from PowerShell in folders with non-ASCII characters, set `$env:PYTHONIOENCODING = "utf-8"`.
