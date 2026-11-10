import json
from urllib.request import Request, urlopen

from fastapi.testclient import TestClient

from api.main import app


PAYLOAD = {
    "PassengerId": 18,
    "Pclass": 2,
    "Name": "Williams, Mr. Charles Eugene",
    "Sex": "male",
    "Age": 32.0,
    "SibSp": 0,
    "Parch": 0,
    "Ticket": "244373",
    "Fare": 13.0,
    "Cabin": None,
    "Embarked": "S",
}


def call_container():
    data = json.dumps(PAYLOAD).encode("utf-8")
    request = Request(
        "http://localhost:8000/predict",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    print("=" * 60)
    print("LOCAL VS CONTAINER PREDICTION CONSISTENCY")
    print("=" * 60)

    with TestClient(app) as client:
        local_response = client.post("/predict", json=PAYLOAD)

    container_response = call_container()

    print("\nLOCAL RESPONSE:")
    print(json.dumps(local_response.json(), indent=2))

    print("\nCONTAINER RESPONSE:")
    print(json.dumps(container_response, indent=2))

    local_data = local_response.json()
    local_prediction = local_data["prediction"]
    container_prediction = container_response["prediction"]

    local_probability = local_data.get("survival_probability")
    container_probability = container_response.get("survival_probability")

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Local prediction      : {local_prediction}")
    print(f"Container prediction  : {container_prediction}")
    print(f"Local probability     : {local_probability}")
    print(f"Container probability : {container_probability}")

    prediction_match = local_prediction == container_prediction
    probability_match = (
        local_probability is not None
        and container_probability is not None
        and abs(local_probability - container_probability) < 1e-10
    )

    print(f"\nPrediction match      : {prediction_match}")
    print(f"Probability match     : {probability_match}")

    if prediction_match and probability_match:
        print("\n[SUCCESS] Local and container predictions are consistent.")
    else:
        print("\n[FAILURE] Local and container predictions differ.")
        raise AssertionError("Prediction consistency check failed.")


if __name__ == "__main__":
    main()
