"""
Lab 6 - Registry Deployment Report.

Reads whichever model version currently holds the "champion" alias and
writes a JSON summary (version, run id, params, metrics) that a deployment
system could consume to know exactly what to ship.
"""
import os
import json
from mlflow.tracking import MlflowClient

MODEL_NAME = "Titanic_Survival_Production_Model"
CHAMPION_ALIAS = "champion"
REPORT_PATH = "artifacts/registry_deployment_report.json"


def generate_registry_report():
    print("[INFO] --- Generating Model Registry Deployment Report ---")
    client = MlflowClient()

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, CHAMPION_ALIAS)
    except Exception as e:
        print(f"[ERROR] Could not find a '{CHAMPION_ALIAS}' version for '{MODEL_NAME}': {e}")
        print("[INFO] Run src/automate_lifecycle.py first.")
        return

    run = client.get_run(champion.run_id)

    report = {
        "model_name": MODEL_NAME,
        "champion_alias": CHAMPION_ALIAS,
        "champion_version": champion.version,
        "run_id": champion.run_id,
        "status": champion.status,
        "source": champion.source,
        "metrics": run.data.metrics,
        "params": run.data.params,
    }

    os.makedirs("artifacts", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Deployment report written to '{REPORT_PATH}'.")
    print(json.dumps(report, indent=4))


if __name__ == "__main__":
    generate_registry_report()
