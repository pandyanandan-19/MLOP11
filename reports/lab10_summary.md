# Lab 10 - ML Model Serving and API Engineering using FastAPI

## 1. Objective

Deploy the trained Titanic Survival ML model as a FastAPI inference service and validate its functionality, input validation, robustness, latency, reliability, and throughput.

---

## 2. Implementation Completed

- FastAPI application (`api/main.py`)
- Health endpoint (`GET /health`)
- Metadata endpoint (`GET /metadata`)
- Deployment bundle loader (`api/model_loader.py`)
- Shared deterministic Titanic cleaning & feature engineering (`api/preprocessing.py`)
- Single prediction endpoint (`POST /predict`)
- Batch prediction endpoint (`POST /predict/batch`)
- Pydantic input/output/error schemas (`api/schemas.py`)
- Structured JSONL inference logging (`api/monitoring_logger.py`)
- Standardized HTTP 422 and HTTP 503/500 error responses
- API functional and robustness test suite (`tests/test_api.py`)
- Performance, reliability, and throughput test suite (`tests/test_performance.py`)

---

## 3. API Endpoints

### `GET /`
Returns basic API service status.

### `GET /health`
Returns `{"status": "healthy"}`.

### `GET /metadata`
Returns model name, URI, service version, and readiness flags (`model_loaded`, `preprocessor_loaded`).

### `POST /predict`
Accepts one passenger record (`PassengerInput`) and returns:
- `prediction` (`0` or `1`)
- `survival_probability` (`float` in `[0.0, 1.0]`)

### `POST /predict/batch`
Accepts `{"passengers": [...]}` (`BatchPredictionRequest`) and returns `count` and `predictions`.

---

## 4. Standardized Error Handling

Validation failures return HTTP 422 responses with:

```json
{
    "status": "error",
    "message": "Request validation failed",
    "details": [
        {
            "field": "Sex",
            "message": "Input should be 'male' or 'female'"
        }
    ]
}
```
