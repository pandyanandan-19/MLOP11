# Lab 7 - ML Pipeline Engineering

Lab 7 builds the core Titanic survival prediction ML pipeline.

Normal stages introduced in Lab 7:

```text
01 Load Data
02 Raw Data Validation
03 Preprocessing
04 Processed Output Validation
05 Training + MLflow + Model Registry
06 Model Evaluation
08 Automated Model Lifecycle
09 Production Inference
```

Lab 8 inserts `07_quality_gate.py` between evaluation and lifecycle, and Lab 13 adds `deploy_if_promoted.py` between lifecycle and production inference.

Run the pipeline after the stages are implemented:

```bash
python src/mlops_pipeline/titanic_survival_pipeline.py --force
```

Training uses a Random Forest classifier (`n_estimators=150`, `max_depth=12`, `class_weight="balanced"`) configured in `config.yaml`. Evaluation is performed on the fixed held-out test split (`test_passenger_ids.csv`) using accuracy, precision, recall, F1, and ROC-AUC. MLflow records parameters, metrics, and the preprocessing pipeline artifact (`models/preprocessor.pkl`), and registers each trained model under:

```text
Titanic_Survival_Production_Model
```

The lifecycle manages stages:

```text
None -> Staging -> Production / Archived
```
