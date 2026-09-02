import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from preprocessing import load_data


RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)


def build_preprocessor(X):
    numeric_features = [
        c for c in [
            "Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"
        ] if c in X.columns
    ]

    categorical_features = [
        c for c in ["Pclass", "Sex", "Embarked", "Title"]
        if c in X.columns
    ]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])


def main():
    X, y, _ = load_data()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            C=1.0,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    }

    results = []

    print("=" * 70)
    print("MODEL TRAINING AND 5-FOLD CROSS VALIDATION")
    print("=" * 70)

    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring="accuracy"
        )

        valid_accuracy = pipeline.score(X_valid, y_valid)

        results.append({
            "Model": name,
            "Validation Accuracy": valid_accuracy,
            "CV Mean Accuracy": cv_scores.mean(),
            "CV Std": cv_scores.std()
        })

        print(f"\n{name}")
        print(f"Validation Accuracy: {valid_accuracy:.4f}")
        print(f"5-Fold CV Mean:      {cv_scores.mean():.4f}")
        print(f"5-Fold CV Std:       {cv_scores.std():.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "CV Mean Accuracy",
        ascending=False
    )

    results_df.to_csv(
        "outputs/results/model_comparison_initial.csv",
        index=False
    )

    print("\nInitial model comparison:")
    print(results_df.to_string(index=False))

    # Hyperparameter tuning for Random Forest
    rf_pipeline = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1
        ))
    ])

    param_grid = {
        "model__n_estimators": [200, 300],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    print("\n" + "=" * 70)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 70)

    grid = GridSearchCV(
        rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    tuned_valid_accuracy = grid.score(X_valid, y_valid)

    print("Best parameters:")
    print(grid.best_params_)
    print(f"Best CV Accuracy: {grid.best_score_:.4f}")
    print(f"Tuned Validation Accuracy: {tuned_valid_accuracy:.4f}")

    # Final model: tuned Random Forest, retrained on all labeled training data
    final_model = grid.best_estimator_
    final_model.fit(X, y)

    joblib.dump(final_model, "models/titanic_model.pkl")

    print("\nFinal model saved to:")
    print("models/titanic_model.pkl")

    # Save final tuning information
    tuning_result = pd.DataFrame([{
        "Model": "Tuned Random Forest",
        "Best CV Accuracy": grid.best_score_,
        "Validation Accuracy": tuned_valid_accuracy,
        "Best Parameters": str(grid.best_params_)
    }])

    tuning_result.to_csv(
        "outputs/results/random_forest_tuning.csv",
        index=False
    )


if __name__ == "__main__":
    main()
