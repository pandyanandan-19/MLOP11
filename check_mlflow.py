import mlflow
import yaml
from mlflow import MlflowClient


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["registry"]["model_name"]
    client = MlflowClient()

    model = client.get_registered_model(model_name)
    print("MODEL:", model.name)
    print("VERSIONS:")

    versions = client.search_model_versions(f"name='{model_name}'")
    versions = sorted(versions, key=lambda v: int(v.version), reverse=True)

    for version in versions:
        print(
            f"version={version.version} "
            f"stage={version.current_stage} "
            f"aliases={version.aliases} "
            f"run_id={version.run_id}"
        )


if __name__ == "__main__":
    main()
