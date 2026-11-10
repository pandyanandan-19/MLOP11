import sys
import yaml
from mlflow import MlflowClient


def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    client = MlflowClient()
    model_name = config["registry"]["model_name"]

    if len(sys.argv) > 1:
        version = sys.argv[1]
    else:
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            raise RuntimeError(f"No versions found for '{model_name}'.")
        prod = [v for v in versions if v.current_stage == "Production"]
        target = prod[0] if prod else sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
        version = target.version

    mv = client.get_model_version(model_name, str(version))
    print("MODEL NAME:", mv.name)
    print("VERSION:", mv.version)
    print("STAGE:", mv.current_stage)
    print("ALIASES:", mv.aliases)
    print("RUN ID:", mv.run_id)
    print("SOURCE:", mv.source)
    print("STATUS:", mv.status)
    print("CREATION TIMESTAMP:", mv.creation_timestamp)


if __name__ == "__main__":
    main()
