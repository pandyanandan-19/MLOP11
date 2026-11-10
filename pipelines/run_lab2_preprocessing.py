import subprocess,sys,os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for s in ["preprocess_pipeline.py","validate_outputs.py"]:
    print(f"[INFO] Running {s}")
    r=subprocess.run([sys.executable,os.path.join(ROOT,"src",s)],cwd=ROOT)
    if r.returncode!=0: raise SystemExit(1)
print("[SUCCESS] Lab 2 Preprocessing pipeline completed.")
