"""
Lab 6 - Model Lifecycle Automation.

Scans every version of the registered model, compares them on the F1 score
logged during training, and promotes the best-performing version by
assigning it the "champion" alias in the MLflow Model Registry.

(Newer MLflow releases replaced the old "Staging/Production" stage labels
with aliases - see https://mlflow.org/docs/latest/model-registry.html)
"""
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "Titanic_Survival_Production_Model"
METRIC_KEY = "f1_score"
CHAMPION_ALIAS = "champion"


def automate_lifecycle():
    print("[INFO] --- Starting Model Lifecycle Automation ---")
    client = MlflowClient()

    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    except Exception as e:
        print(f"[ERROR] Could not query registered model '{MODEL_NAME}': {e}")
        print("[INFO] Has src/train_registry.py been run yet?")
        return

    if not versions:
        print(f"[ERROR] No versions found for model '{MODEL_NAME}'. "
              f"Run src/train_registry.py first.")
        return

    print(f"[INFO] Found {len(versions)} version(s) of '{MODEL_NAME}'.")

    best_version = None
    best_score = float("-inf")

    for v in versions:
        run = client.get_run(v.run_id)
        score = run.data.metrics.get(METRIC_KEY)

        if score is None:
            print(f"  - Version {v.version}: no '{METRIC_KEY}' metric found, skipping.")
            continue

        print(f"  - Version {v.version}: {METRIC_KEY} = {score:.4f}")

        if score > best_score:
            best_score = score
            best_version = v.version

    if best_version is None:
        print(f"[ERROR] None of the versions had a usable '{METRIC_KEY}' metric. Nothing promoted.")
        return

    print(f"[INFO] Promoting version {best_version} (F1 = {best_score:.4f}) "
          f"to alias '{CHAMPION_ALIAS}'...")
    client.set_registered_model_alias(MODEL_NAME, CHAMPION_ALIAS, best_version)

    print(f"[SUCCESS] Version {best_version} is now the '{CHAMPION_ALIAS}' model "
          f"(F1 = {best_score:.4f}).")


if __name__ == "__main__":
    automate_lifecycle()
