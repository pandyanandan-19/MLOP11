import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE_REPORT = PROJECT_ROOT / "reports" / "lifecycle_report.json"
PREPARE_SCRIPT = PROJECT_ROOT / "src" / "mlops_pipeline" / "11_prepare_deployment.py"
DEPLOY_SCRIPT = PROJECT_ROOT / "deployment" / "deploy_production.py"
DEPLOYMENT_REPORT = PROJECT_ROOT / "reports" / "deployment_automation_report.json"
METADATA_PATH = PROJECT_ROOT / "deployment" / "model_metadata.json"


def load_lifecycle_report():
    if not LIFECYCLE_REPORT.exists():
        raise FileNotFoundError(f"Lifecycle report not found: {LIFECYCLE_REPORT}")

    with LIFECYCLE_REPORT.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_script(script_path):
    result = subprocess.run([sys.executable, str(script_path)], cwd=PROJECT_ROOT)
    return result.returncode == 0


def is_command_ready(command, timeout=10):
    """Return True when the external CLI command succeeds within the timeout."""
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def record_bundle_only_report(reason):
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    DEPLOYMENT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "bundle_prepared_only",
        "reason": reason,
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "run_id": metadata["run_id"],
        "bundle_prepared": True,
        "deployment_bundle": {
            "model": "deployment/model.pkl",
            "preprocessor": "deployment/preprocessor.pkl",
            "metadata": "deployment/model_metadata.json",
        },
    }
    with DEPLOYMENT_REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)


def main():
    print("=" * 70)
    print("DEPLOY IF MODEL WAS PROMOTED")
    print("=" * 70)

    report = load_lifecycle_report()

    promoted = report.get("promoted", False)
    candidate_version = report.get("candidate_version")
    production_version = report.get("production_version_after")

    print(f"[INFO] Candidate version : {candidate_version}")
    print(f"[INFO] Promoted          : {promoted}")
    print(f"[INFO] Production version: {production_version}")

    if not promoted:
        print("[INFO] Model was not promoted. Deployment skipped.")
        return True

    print(f"[INFO] Model Version {candidate_version} was promoted.")
    print("[INFO] Preparing Production deployment artifacts...")

    if not run_script(PREPARE_SCRIPT):
        print("[ERROR] Deployment preparation failed.")
        return False

    print("[SUCCESS] Deployment artifacts prepared.")

    docker_ready = shutil.which("docker") is not None and is_command_ready(["docker", "info"])
    kubectl_ready = shutil.which("kubectl") is not None and is_command_ready(
        ["kubectl", "cluster-info", "--request-timeout=5s"]
    )

    if not docker_ready or not kubectl_ready:
        print(
            "[WARN] Docker daemon or Kubernetes cluster not available; "
            "skipping container/Kubernetes rollout."
        )
        record_bundle_only_report(
            "docker daemon or kubernetes cluster not reachable in current environment"
        )
        return True

    print("[INFO] Starting Production deployment...")
    if not run_script(DEPLOY_SCRIPT):
        print("[ERROR] Production deployment failed.")
        return False

    print("[SUCCESS] Production deployment completed.")
    return True


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
