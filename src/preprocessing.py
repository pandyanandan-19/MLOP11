import pandas as pd

DROP_COLUMNS = ["PassengerId", "Name", "Ticket", "Cabin"]

def add_features(df):
    df = df.copy()

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    df["Title"] = (
        df["Name"]
        .str.extract(r",\s*([^.]*)\.", expand=False)
        .str.strip()
    )

    # Reduce rare titles to a common category
    common_titles = ["Mr", "Miss", "Mrs", "Master"]
    df["Title"] = df["Title"].where(
        df["Title"].isin(common_titles), "Rare"
    )

    return df


def prepare_features(df):
    df = add_features(df)

    columns_to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    df = df.drop(columns=columns_to_drop)

    return df


def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    X = train.drop("Survived", axis=1)
    y = train["Survived"]

    X = prepare_features(X)
    test_prepared = prepare_features(test)

    return X, y, test_prepared
