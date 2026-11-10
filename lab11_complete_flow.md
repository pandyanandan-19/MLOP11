# Lab 11 - Containerization and Deployment of ML Systems

## Purpose

This document tracks the containerization, API consistency, failure recovery, and Kubernetes deployment architecture for the Titanic Survival Prediction MLOps project.

---

# 1. Starting Point (Lab 10 Baseline)

- FastAPI application (`api/main.py`)
- Endpoints: `/`, `/health`, `/metadata`, `/predict`, `/predict/batch`
- Pydantic validation schemas (`api/schemas.py`)
- Shared deterministic Titanic feature engineering (`api/preprocessing.py`)
- Structured JSONL inference logging (`api/monitoring_logger.py`)
- Automated test suites:
  - `tests/test_api.py` (24 passed)
  - `tests/test_performance.py` (6 passed)

---

# 2. Target Architecture

```text
Existing ML Pipeline
        |
        v
MLflow Production Model
        |
        v
Deployment Bundle (deployment/)
        |
        v
FastAPI (/predict, /predict/batch)
        |
        v
Docker Image (titanic-survival-api)
        |
        v
Kubernetes Deployment (titanic-api)
        |
        v
Kubernetes NodePort Service (titanic-api-service:8000)
```

---

# 3. Stage Summary

| Stage | Component | Artifact / File |
|---|---|---|
| 11.1-11.2 | Bundle packaging from MLflow | `src/mlops_pipeline/11_prepare_deployment.py`, `deployment/` |
| 11.3-11.4 | Container definition | `Dockerfile`, `.dockerignore`, `requirements-api.txt` |
| 11.5-11.8 | Container execution & parity check | `tests/test_local_vs_container.py` |
| 11.10-11.12 | Failure simulation & immutable recovery | `docs/lab11_deployment_debugging_report.md` |
| 11.13-11.17 | Kubernetes manifests & rollout automation | `k8s/deployment.yaml`, `k8s/service.yaml`, `deployment/deploy_production.py` |
| 11.18-11.20 | Containerization & debugging documentation | `docs/lab11_containerization_report.md`, `lab11_commands/` |
