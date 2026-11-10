# Lab 11 - Deployment and Debugging Guide

## 1. Objective

Document failure detection, log-based root-cause analysis, immutable container recovery, and Kubernetes pod self-healing procedures for the Titanic Survival API (`titanic-api`).

---

## 2. Missing Deployment Artifact Failure Mode

`api/model_loader.py` verifies `deployment/model.pkl`, `deployment/preprocessor.pkl`, and `deployment/model_metadata.json` during FastAPI `lifespan` startup:

1. If `deployment/model.pkl` is missing or unreadable, `ModelLoader.load_deployment_bundle()` raises `FileNotFoundError` and the container exits immediately rather than serving uninitialized predictions.
2. If an error occurs after startup (`model_loader.is_ready()` is `False`), `POST /predict` and `POST /predict/batch` return standardized HTTP `503` responses and log failure events to `logs/inference/inference_log.jsonl`.

---

## 3. Docker Failure Inspection & Recovery Commands

```powershell
# Inspect exited container status and exit code
docker ps -a --filter "name=titanic-api"
docker inspect titanic-api --format "{{.State.Status}} | ExitCode={{.State.ExitCode}}"

# Inspect application traceback from container logs
docker logs --tail 30 titanic-api

# Recover cleanly by recreating from the immutable image
docker rm titanic-api
docker run --name titanic-api -p 8000:8000 titanic-survival-api:1.0
```

---

## 4. Kubernetes Scaling & Self-Healing Commands

```powershell
# Deploy manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Scale to 3 replicas
kubectl scale deployment titanic-api --replicas=3
kubectl get deployments
kubectl get pods

# Simulate pod interruption and observe replacement pod creation
kubectl delete pod <pod-name>
kubectl get pods -w
kubectl get endpoints titanic-api-service

# Validate recovered service via port-forward
kubectl port-forward service/titanic-api-service 8000:8000
```

---

## 5. Verification Status

- Local `ModelLoader` readiness, standardized `422` / `503` / `500` error handling, and `logs/inference/inference_log.jsonl` logging are verified via `api/main.py` and `tests/test_api.py`.
- Live Docker and Kubernetes failure-injection commands above require Docker Desktop and `kubectl` on the host machine.
