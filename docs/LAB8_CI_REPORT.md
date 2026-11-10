# Lab 8 - Pipeline Automation and CI Workflows

## Goal

Automate the Titanic Survival MLOps pipeline with GitHub Actions (`.github/workflows/lab8_ci.yml`). Lab 8 executes the integrated pipeline and API/performance tests whenever relevant repository changes occur.

## CI flow

```text
Git push / pull request / manual trigger
                |
                v
        install dependencies
                |
                v
titanic_survival_pipeline.py --force
                |
                v
01 -> 02 -> 03 -> 04 -> 05 -> 06 -> 07 -> 08 -> deploy_if_promoted -> 09
                |
                v
    pytest tests/test_api.py tests/test_performance.py
                |
                v
        upload reports/artifacts
```

## Triggered changes

- `config.yaml`
- `src/mlops_pipeline/**`
- `requirements.txt`
- `data/raw/titanic.csv`
- `.github/workflows/lab8_ci.yml`

## Quality gate

```yaml
quality_gate:
  metric: recall
  minimum_value: 0.70
```

If recall is below the threshold, the pipeline exits before lifecycle promotion.

## Jenkins comparison

| Feature | GitHub Actions | Jenkins |
|---|---|---|
| Repository integration | Native to GitHub | Requires integration |
| Workflow definition | YAML | Jenkinsfile / UI |
| Hosted runners | Available | Normally self-managed |
| Setup for this lab | Simple | More infrastructure |
| Use in this project | Implemented | Compared conceptually |

## MLflow note

For CI runs to share one experiment history across temporary GitHub runners, set the GitHub Actions secret `MLFLOW_TRACKING_URI` to a persistent MLflow Tracking Server. See [`MLFLOW_CI_SETUP.md`](MLFLOW_CI_SETUP.md).
