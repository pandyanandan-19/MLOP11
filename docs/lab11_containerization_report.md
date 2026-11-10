# Lab 11 - Containerization and Kubernetes Deployment Report

## 1. Objective

Package the Titanic Survival Prediction FastAPI service into a portable container image, verify the MLflow-backed deployment bundle, and define Kubernetes Deployment and Service manifests alongside automated rollout and consistency verification scripts.

---

## 2. Deployment Architecture

```text
MLflow Registry (Titanic_Survival_Production_Model, Stage=Production)
        |
        v
11_prepare_deployment.py
        |
        v
deployment/
  |-- model.pkl
  |-- preprocessor.pkl
  \-- model_metadata.json
        |
        v
Dockerfile (python:3.11-slim + requirements-api.txt)
        |
        v
k8s/deployment.yaml (titanic-api) + k8s/service.yaml (titanic-api-service)
        |
        v
FastAPI Service (:8000)
```

MLflow is not required during API container runtime because `11_prepare_deployment.py` extracts the verified Production model and matching preprocessor from the same MLflow run into `deployment/`.

---

## 3. Verified Deployment Bundle

- **Registered Model**: `Titanic_Survival_Production_Model`
- **Production Version**: `1`
- **Stage**: `Production`
- **Run ID**: `737141f08de64811a0d7882185647d10`
- **Model ID**: `m-68dfada5c17846738374941650bc7314`
- **Files Packaged**:
  - `deployment/model.pkl`
  - `deployment/preprocessor.pkl`
  - `deployment/model_metadata.json`

---

## 4. Container & Kubernetes Configuration

- **`Dockerfile`**: uses `python:3.11-slim`, installs lean API dependencies from `requirements-api.txt`, copies `api/`, `deployment/`, and `config.yaml`, exposes port `8000`, and starts `uvicorn api.main:app --host 0.0.0.0 --port 8000`.
- **`.dockerignore`**: excludes virtual environments, `.git/`, `.github/`, `mlruns/`, `mlflow.db`, `notebooks/`, `tests/`, `artifacts/`, `logs/`, `outputs/`, `reports/`, and `models/`.
- **`k8s/deployment.yaml`**: defines `Deployment/titanic-api` with `titanic-survival-api:1.0`, port `8000`, `imagePullPolicy: IfNotPresent`.
- **`k8s/service.yaml`**: defines `NodePort` service `titanic-api-service` forwarding port `8000`.
- **`deployment/deploy_production.py`**: automates Docker image inspection/build (`titanic-survival-api:model-v<version>`), `kubectl set image`, `kubectl rollout status --timeout=120s`, and writes `reports/deployment_automation_report.json`.
- **`tests/test_local_vs_container.py`**: compares local `TestClient(app)` predictions and `survival_probability` against `http://localhost:8000/predict`.

---

## 5. Execution Scope Note

- `src/mlops_pipeline/11_prepare_deployment.py` and the local FastAPI `TestClient` suites (`tests/test_api.py` and `tests/test_performance.py`) were executed and verified locally.
- Docker image builds and Kubernetes cluster deployments (`docker build`, `kubectl apply`, `deployment/deploy_production.py`, `tests/test_local_vs_container.py`) require Docker Desktop and `kubectl` on the target host and were not run in this environment because `docker` and `kubectl` are not installed here.
