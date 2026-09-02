# Titanic Survival Prediction Using Machine Learning

## 1. Project Overview

This project develops an end-to-end machine learning classification system for predicting whether a Titanic passenger survived.

## 2. Problem Statement

The objective is to predict passenger survival using passenger information such as passenger class, sex, age, family information, fare and port of embarkation.

## 3. Machine Learning Type

- Learning: Supervised Learning
- Problem: Binary Classification
- Target: `Survived`
- 0 = Did Not Survive
- 1 = Survived

## 4. Dataset

The project uses the Kaggle Titanic dataset:

- `train.csv`: labeled training data
- `test.csv`: unseen passenger data
- `gender_submission.csv`: sample submission format

## 5. Project Workflow

1. Understand the dataset
2. Exploratory Data Analysis
3. Missing value analysis
4. Duplicate analysis
5. Outlier analysis
6. Feature engineering
7. Feature selection
8. Train-validation split
9. Preprocessing pipeline
10. Multiple ML models
11. 5-fold cross-validation
12. Hyperparameter tuning
13. Final model evaluation
14. Best model selection
15. Unseen passenger prediction
16. Test-set prediction file

## 6. Feature Engineering

Created features:

- `FamilySize = SibSp + Parch + 1`
- `IsAlone`
- `Title` extracted from passenger names

## 7. Preprocessing

Numerical features:
- Median imputation
- Standard scaling

Categorical features:
- Most-frequent imputation
- One-hot encoding

## 8. Models

The project compares:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors
- Support Vector Machine
- Gradient Boosting

Random Forest is additionally tuned using GridSearchCV.

## 9. Evaluation

Metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- 5-fold Cross Validation

## 10. Output Files

Generated files include:

- `models/titanic_model.pkl`
- `outputs/results/model_comparison_initial.csv`
- `outputs/results/random_forest_tuning.csv`
- `outputs/results/final_model_metrics.csv`
- `outputs/results/titanic_predictions.csv`

## 11. How to Run

Activate the virtual environment and install requirements.

Then run:

```bash
python src/data_analysis.py
python src/eda.py
python src/train_models.py
python src/evaluate.py
python src/generate_predictions.py
python src/predict.py
```

## 12. Final Conclusion

The final model is selected based on actual validation, cross-validation and evaluation results. The selected model is then retrained on the complete labeled training dataset and used to generate predictions for unseen passengers.
