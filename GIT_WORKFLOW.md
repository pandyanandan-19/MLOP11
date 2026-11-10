# Git Workflow Documentation - Titanic Survival MLOps

**Project:** Titanic_Survival_ML (Titanic Passenger Survival MLOps Pipeline)
**Scope:** Version control, modular branching, validation-gated commits, and CI triggers

---

## 1. Repository Overview

The repository is structured around modular pipeline stages separating:
- `src/` (Labs 1-6 scripts for validation, preprocessing, training, evaluation, and registry reporting)
- `src/mlops_pipeline/` (Labs 7-13 end-to-end orchestration, quality gate, lifecycle, failure testing, drift monitoring, and continuous retraining)
- `api/` (FastAPI model serving layer)
- `deployment/` and `k8s/` (containerized deployment automation and Kubernetes manifests)
- `tests/` (functional, robustness, performance, and container consistency tests)
- `.github/workflows/` (CI, failure testing, runner verification, and monitoring automation)

---

## 2. Recommended Branching Strategy

- `main`: protected integration branch holding verified code, `config.yaml`, raw baseline data (`data/raw/titanic.csv`), and workflow definitions.
- `feature/*`: short-lived branches for feature engineering, model configuration tuning, or API changes.
- `lab13-integration`: integration verification branch used with `.github/workflows/lab13_runner_test.yml`.

---

## 3. Pre-Commit Validation Checklist

Before merging changes into `main`:

1. Validate raw data schema:
   ```bash
   python src/mlops_pipeline/02_validate_data.py
   ```
2. Run the end-to-end pipeline:
   ```bash
   python src/mlops_pipeline/titanic_survival_pipeline.py --force
   ```
3. Run failure and recovery checks:
   ```bash
   python src/mlops_pipeline/10_failure_injection.py
   ```
4. Run API and performance tests:
   ```bash
   pytest tests/test_api.py tests/test_performance.py -v
   ```

---

## 4. Rollback & Fault Detection Policy

- Never commit runtime databases (`mlflow.db`), generated `.pkl` binaries, or transient `outputs/` / `logs/` artifacts (see `.gitignore`).
- If a preprocessing or schema regression is introduced, `02_validate_data.py` and `04_validate_outputs.py` fail fast before training executes.
- Use `git revert <commit>` rather than `git reset --hard` on shared branches to preserve an auditable history.
