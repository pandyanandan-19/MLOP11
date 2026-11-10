import pytest
from starlette.testclient import TestClient

from api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


VALID_PASSENGER = {
    "PassengerId": 1,
    "Pclass": 3,
    "Name": "Braund, Mr. Owen Harris",
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "A/5 21171",
    "Fare": 7.25,
    "Cabin": None,
    "Embarked": "S",
}


SECOND_PASSENGER = {
    "PassengerId": 2,
    "Pclass": 1,
    "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
    "Sex": "female",
    "Age": 38.0,
    "SibSp": 1,
    "Parch": 0,
    "Ticket": "PC 17599",
    "Fare": 71.2833,
    "Cabin": "C85",
    "Embarked": "C",
}


# ---------------------------------------------------------
# 9.1 - Health Endpoint
# ---------------------------------------------------------

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


# ---------------------------------------------------------
# 9.2 - Metadata Endpoint
# ---------------------------------------------------------

def test_metadata(client):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()

    assert data["service"] == "Titanic Survival Prediction API"
    assert data["version"] == "1.0.0"
    assert data["model_loaded"] is True
    assert data["preprocessor_loaded"] is True
    assert data["model_name"] is not None
    assert data["model_uri"] is not None


# ---------------------------------------------------------
# 9.3 - Valid Single Prediction
# ---------------------------------------------------------

def test_valid_prediction(client):
    response = client.post("/predict", json=VALID_PASSENGER)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert "survival_probability" in data
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["survival_probability"] <= 1.0


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Sex
# ---------------------------------------------------------

def test_invalid_prediction_sex(client):
    payload = VALID_PASSENGER.copy()
    payload["Sex"] = "ABC"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "Sex"


# ---------------------------------------------------------
# 9.4 - Invalid Single Prediction: Age
# ---------------------------------------------------------

def test_invalid_prediction_age(client):
    payload = VALID_PASSENGER.copy()
    payload["Age"] = -5.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "Age"


# ---------------------------------------------------------
# 9.4 - Missing Required Field
# ---------------------------------------------------------

def test_missing_prediction_field(client):
    payload = VALID_PASSENGER.copy()
    del payload["Pclass"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "Pclass"


# ---------------------------------------------------------
# 9.5 - Valid Batch Prediction
# ---------------------------------------------------------

def test_valid_batch_prediction(client):
    payload = {"passengers": [VALID_PASSENGER, SECOND_PASSENGER]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 2
    assert len(data["predictions"]) == 2

    for prediction in data["predictions"]:
        assert "prediction" in prediction
        assert "survival_probability" in prediction
        assert prediction["prediction"] in [0, 1]
        assert 0.0 <= prediction["survival_probability"] <= 1.0


# ---------------------------------------------------------
# 9.6 - Invalid Batch Prediction
# ---------------------------------------------------------

def test_invalid_batch_prediction(client):
    invalid_passenger = SECOND_PASSENGER.copy()
    invalid_passenger["Sex"] = "ABC"

    payload = {"passengers": [VALID_PASSENGER, invalid_passenger]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "passengers.1.Sex"


# ---------------------------------------------------------
# 9.7 - Multiple Batch Validation Errors
# ---------------------------------------------------------

def test_multiple_batch_validation_errors(client):
    first_passenger = VALID_PASSENGER.copy()
    second_passenger = SECOND_PASSENGER.copy()

    first_passenger["Sex"] = "ABC"
    second_passenger["Age"] = -5.0

    payload = {"passengers": [first_passenger, second_passenger]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 2

    fields = [error["field"] for error in data["details"]]
    assert "passengers.0.Sex" in fields
    assert "passengers.1.Age" in fields


# ---------------------------------------------------------
# 9.8 - Empty Batch
# ---------------------------------------------------------

def test_empty_batch(client):
    payload = {"passengers": []}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "passengers"


# ---------------------------------------------------------
# 9.9 - Invalid Datatype
# ---------------------------------------------------------

def test_invalid_datatype(client):
    payload = VALID_PASSENGER.copy()
    payload["SibSp"] = "two"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "SibSp"


# ---------------------------------------------------------
# 9.10 - Extreme Valid Numeric Values & Nullable Fields
# ---------------------------------------------------------

def test_extreme_numeric_values(client):
    payload = VALID_PASSENGER.copy()
    payload["Age"] = 100.0
    payload["Fare"] = 10000.0
    payload["SibSp"] = 10
    payload["Parch"] = 10

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["survival_probability"] <= 1.0


def test_nullable_optional_fields(client):
    payload = VALID_PASSENGER.copy()
    payload["Age"] = None
    payload["Fare"] = None
    payload["Cabin"] = None
    payload["Embarked"] = None

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["survival_probability"] <= 1.0


# =========================================================
# STAGE 10 - INVALID / EXTREME / CORRUPTED INPUT TESTING
# =========================================================

def test_null_value(client):
    payload = VALID_PASSENGER.copy()
    payload["Sex"] = None

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()

    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "Sex"


def test_malformed_json(client):
    malformed_json = """
    {
        "PassengerId": 1,
        "Sex": "male"
    """

    response = client.post(
        "/predict",
        content=malformed_json,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) >= 1


def test_wrong_json_structure(client):
    payload = ["this", "should", "be", "an", "object"]

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) >= 1


def test_null_numeric_value(client):
    payload = VALID_PASSENGER.copy()
    payload["SibSp"] = None

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "SibSp"


def test_null_pclass(client):
    payload = VALID_PASSENGER.copy()
    payload["Pclass"] = None

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) == 1
    assert data["details"][0]["field"] == "Pclass"


def test_multiple_invalid_datatypes(client):
    payload = VALID_PASSENGER.copy()
    payload["Age"] = "thirty"
    payload["SibSp"] = "two"
    payload["Pclass"] = "first"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) >= 2

    fields = [error["field"] for error in data["details"]]
    assert "Age" in fields
    assert "SibSp" in fields


def test_invalid_batch_structure(client):
    payload = {"passengers": VALID_PASSENGER}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) >= 1


def test_null_batch_passengers(client):
    payload = {"passengers": None}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) >= 1


def test_batch_with_null_passenger(client):
    payload = {"passengers": [VALID_PASSENGER, None]}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"
    assert len(data["details"]) >= 1


def test_extreme_negative_values(client):
    payload = VALID_PASSENGER.copy()
    payload["Age"] = -1000.0
    payload["Fare"] = -9999.0

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert data["status"] == "error"
    assert data["message"] == "Request validation failed"

    fields = [error["field"] for error in data["details"]]
    assert "Age" in fields
    assert "Fare" in fields


def test_large_batch(client):
    passengers = []

    for i in range(1, 21):
        passenger = VALID_PASSENGER.copy()
        passenger["PassengerId"] = i
        passengers.append(passenger)

    payload = {"passengers": passengers}

    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["count"] == 20
    assert len(data["predictions"]) == 20

    for prediction in data["predictions"]:
        assert prediction["prediction"] in [0, 1]
        assert 0.0 <= prediction["survival_probability"] <= 1.0
