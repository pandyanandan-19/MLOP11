import subprocess,sys,os
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for s in ["validate_data.py"]:
    r=subprocess.run([sys.executable,os.path.join(ROOT,"src",s)],cwd=ROOT)
    if r.returncode!=0: raise SystemExit(1)
print("[SUCCESS] Lab 1 Data Validation pipeline completed.")
