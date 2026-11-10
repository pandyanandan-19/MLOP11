import sys
import yaml
from mlflow import MlflowClient


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    if len(sys.argv) > 1:
        run_id = sys.argv[1]
    else:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"No versions found for '{model_name}'.")
        prod = [v for v in versions if v.current_stage == "Production"]
        target = prod[0] if prod else sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        run_id = target.run_id

    run = client.get_run(run_id)
    print("RUN ID:", run.info.run_id)
    print("STATUS:", run.info.status)
    print("ARTIFACT URI:", run.info.artifact_uri)
    print("\nARTIFACTS:")

    artifacts = client.list_artifacts(run_id)
    for artifact in artifacts:
        print(f"{artifact.path} | is_dir={artifact.is_dir}")
        if artifact.is_dir:
            children = client.list_artifacts(run_id, artifact.path)
            for child in children:
                print(f"  +-- {child.path} | is_dir={child.is_dir}")


if __name__ == "__main__":
    main()
