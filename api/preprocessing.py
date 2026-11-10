import numpy as np
import pandas as pd


COMMON_TITLES = ["Mr", "Miss", "Mrs", "Master"]
NUM_COLS = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
STR_CAT_COLS = ["Sex", "Embarked", "Title"]
INT_CAT_COLS = ["Pclass", "IsAlone"]
DROP_COLS = ["PassengerId", "Name", "Ticket", "Cabin", "Survived"]


def clean_prediction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic Titanic feature engineering and cleaning before preprocessing."""
    df = df.copy()

    if "Title" not in df.columns:
        if "Name" in df.columns:
            extracted = (
                df["Name"]
                .astype("string")
                .str.extract(r",\s*([^.]*)\.", expand=False)
                .str.strip()
            )
            df["Title"] = extracted.where(extracted.isin(COMMON_TITLES), "Rare").fillna("Rare")
        else:
            df["Title"] = "Rare"

    if "FamilySize" not in df.columns and {"SibSp", "Parch"}.issubset(df.columns):
        sibsp = pd.to_numeric(df["SibSp"], errors="coerce").fillna(0).astype(int)
        parch = pd.to_numeric(df["Parch"], errors="coerce").fillna(0).astype(int)
        df["FamilySize"] = sibsp + parch + 1

    if "IsAlone" not in df.columns and "FamilySize" in df.columns:
        df["IsAlone"] = (pd.to_numeric(df["FamilySize"], errors="coerce") == 1).astype(int)

    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in INT_CAT_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in STR_CAT_COLS:
        if col in df.columns:
            series = df[col].astype(object)
            mask = series.notna() & (series.astype(str).str.strip() != "")
            df[col] = series.where(mask, np.nan)

    cols_to_drop = [col for col in DROP_COLS if col in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    return df
