import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def run_preprocessing():
    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load the raw data
    data_path = 'data/raw/titanic.csv'
    if not os.path.exists(data_path):
        data_path = 'data/train.csv'
    df = pd.read_csv(data_path)

    # 2. Basic Data Formatting and Feature Engineering
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        df['Title'] = df['Title'].where(df['Title'].isin(common_titles), 'Rare')
    else:
        df['Title'] = 'Unknown'

    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Target conversion
    df['Survived'] = df['Survived'].astype(int)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 3. Train-Test Split (Done before transformations to prevent data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Separate Column Types
    num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    # 5. Build Modular Scikit-Learn Pipelines
    # Numerical pipeline: Median imputation, then standard scaling
    num_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Mode imputation, then One-Hot Encoding
    cat_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])

    # Combine into a single ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )

    # 6. Execute Pipeline
    print("[INFO] Fitting and transforming training data...")
    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")
    X_test_final = preprocessor.transform(X_test)

    # 7. Save Artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))

    # Save the single unified preprocessor artifact
    joblib.dump(preprocessor, 'models/preprocessor.pkl')

    # Save Metadata
    metadata = {
        "dataset_name": "Titanic Passenger Survival",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully! Pipeline saved as preprocessor.pkl.")


if __name__ == "__main__":
    run_preprocessing()
