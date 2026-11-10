import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
METADATA_PATH = DEPLOYMENT_DIR / "model_metadata.json"
K8S_DEPLOYMENT_PATH = PROJECT_ROOT / "k8s" / "deployment.yaml"
REPORT_PATH = PROJECT_ROOT / "reports" / "deployment_automation_report.json"

IMAGE_NAME = "titanic-survival-api"


def run_command(command):
    print(f"[INFO] Running: {' '.join(command)}")
    result = subprocess.run(command, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(command)}")

    return result


def load_metadata():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Deployment metadata not found: {METADATA_PATH}")

    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_kubernetes_config():
    if not K8S_DEPLOYMENT_PATH.exists():
        raise FileNotFoundError(f"Kubernetes deployment file not found: {K8S_DEPLOYMENT_PATH}")

    with K8S_DEPLOYMENT_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    deployment_name = config["metadata"]["name"]
    app_label = config["metadata"]["labels"]["app"]
    container_name = config["spec"]["template"]["spec"]["containers"][0]["name"]

    return deployment_name, app_label, container_name


def save_report(report):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def deploy_production(local_only=False):
    print("=" * 70)
    print("PRODUCTION DEPLOYMENT")
    print("=" * 70)

    metadata = load_metadata()
    deployment_name, app_label, container_name = load_kubernetes_config()

    model_name = metadata["model_name"]
    model_version = metadata["model_version"]
    model_stage = metadata["model_stage"]
    run_id = metadata["run_id"]

    if model_stage != "Production":
        raise RuntimeError(f"Deployment blocked. Model stage is '{model_stage}', not 'Production'.")

    model_path = DEPLOYMENT_DIR / "model.pkl"
    preprocessor_path = DEPLOYMENT_DIR / "preprocessor.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessor artifact not found: {preprocessor_path}")

    # Verify artifacts load cleanly before proceeding
    joblib.load(model_path)
    joblib.load(preprocessor_path)

    image_tag = f"{IMAGE_NAME}:model-v{model_version}"

    print(f"[INFO] Model        : {model_name}")
    print(f"[INFO] Model version: {model_version}")
    print(f"[INFO] Run ID       : {run_id}")
    print(f"[INFO] Docker image : {image_tag}")
    print(f"[INFO] Kubernetes   : {deployment_name}")
    print(f"[INFO] Container    : {container_name}")

    has_docker = shutil.which("docker") is not None
    has_kubectl = shutil.which("kubectl") is not None

    if local_only:
        report = {
            "status": "local_bundle_verified",
            "model_name": model_name,
            "model_version": str(model_version),
            "run_id": run_id,
            "local_api_url": "http://127.0.0.1:8000/docs",
            "deployment_bundle": {
                "model": str(model_path.relative_to(PROJECT_ROOT)),
                "preprocessor": str(preprocessor_path.relative_to(PROJECT_ROOT)),
                "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
            },
        }
        save_report(report)
        print("[SUCCESS] Deployment bundle verified for local FastAPI/Uvicorn serving.")
        print("[INFO] Start API with: .\\.venv\\Scripts\\uvicorn.exe api.main:app --host 127.0.0.1 --port 8000")
        print("[INFO] Open in browser: http://127.0.0.1:8000/docs")
        return True

    if not has_docker or not has_kubectl:
        missing = [tool for tool, ok in (("docker", has_docker), ("kubectl", has_kubectl)) if not ok]
        raise RuntimeError(
            f"Missing required CLI tool(s): {', '.join(missing)}. "
            "Install Docker Desktop with Kubernetes enabled (or run with --local for local Uvicorn serving)."
        )

    image_exists = (
        subprocess.run(
            ["docker", "image", "inspect", image_tag],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )

    if image_exists:
        print(f"[INFO] Docker image already exists: {image_tag}")
        print("[INFO] Reusing existing image.")
    else:
        print("[INFO] Docker image not found. Building...")
        run_command(["docker", "build", "-t", image_tag, "."])

    print("[INFO] Updating Kubernetes deployment...")
    run_command([
        "kubectl",
        "set",
        "image",
        f"deployment/{deployment_name}",
        f"{container_name}={image_tag}",
    ])

    print("[INFO] Waiting for Kubernetes rollout...")
    run_command([
        "kubectl",
        "rollout",
        "status",
        f"deployment/{deployment_name}",
        "--timeout=120s",
    ])

    report = {
        "status": "success",
        "model_name": model_name,
        "model_version": str(model_version),
        "run_id": run_id,
        "docker_image": image_tag,
        "kubernetes_context": "docker-desktop",
        "kubernetes_deployment": deployment_name,
        "kubernetes_container": container_name,
        "app_label": app_label,
        "deployment_bundle": {
            "model": str(model_path.relative_to(PROJECT_ROOT)),
            "preprocessor": str(preprocessor_path.relative_to(PROJECT_ROOT)),
            "metadata": str(METADATA_PATH.relative_to(PROJECT_ROOT)),
        },
    }

    save_report(report)

    print("[SUCCESS] Docker image built.")
    print("[SUCCESS] Kubernetes deployment updated.")
    print("[SUCCESS] Kubernetes rollout completed.")
    print(f"[SUCCESS] Deployment report saved to {REPORT_PATH.relative_to(PROJECT_ROOT)}")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Titanic Survival API to Kubernetes or verify local bundle.")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Verify deployment bundle for local Uvicorn serving when Docker/Kubernetes are not installed.",
    )
    args = parser.parse_args()

    try:
        passed = deploy_production(local_only=args.local)
        sys.exit(0 if passed else 1)
    except Exception as exc:
        print(f"[ERROR] Production deployment failed: {exc}")
        sys.exit(1)
