import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def run_preprocessing():
    print("Starting Preprocessing Pipeline...")

    # 1. Load the raw data
    data_path = 'data/raw/titanic.csv'
    if not os.path.exists(data_path):
        data_path = 'data/train.csv'
    df = pd.read_csv(data_path)

    # 2. Data Cleaning & Feature Engineering
    # Extract Title from Name
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        df['Title'] = df['Title'].where(df['Title'].isin(common_titles), 'Rare')
    else:
        df['Title'] = 'Unknown'

    # Family Size and IsAlone features
    if 'SibSp' in df.columns and 'Parch' in df.columns:
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Drop high-cardinality non-predictive identifiers
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Handle missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    if 'Embarked' in df.columns:
        df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Fare' in df.columns:
        df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Target binary integer conversion
    df['Survived'] = df['Survived'].astype(int)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    # 3. Train-Test Split (stratified on target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Separate Column Types
    num_cols = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']
    num_cols = [c for c in num_cols if c in X_train.columns]
    cat_cols = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone']
    cat_cols = [c for c in cat_cols if c in X_train.columns]

    # 5. Scale & Encode
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train[num_cols])
    x_test_scaled = scaler.transform(X_test[num_cols])

    ohe = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    x_train_encoded = ohe.fit_transform(X_train[cat_cols])
    x_test_encoded = ohe.transform(X_test[cat_cols])

    # Combine Features
    X_train_final = np.hstack((x_train_scaled, x_train_encoded))
    X_test_final = np.hstack((x_test_scaled, x_test_encoded))

    # 6. Save Artifacts
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    np.save('data/processed/X_train_final.npy', X_train_final)
    np.save('data/processed/X_test_final.npy', X_test_final)
    np.save('data/processed/y_train.npy', y_train.to_numpy(dtype=np.int64))
    np.save('data/processed/y_test.npy', y_test.to_numpy(dtype=np.int64))

    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(ohe, 'models/ohe.pkl')

    # Save Metadata
    metadata = {
        "dataset_name": "Titanic Passenger Survival",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols)
    }
    with open('data/processed/dataset_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)

    print("Preprocessing completed successfully!")


if __name__ == "__main__":
    run_preprocessing()
