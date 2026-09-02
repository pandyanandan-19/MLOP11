import pandas as pd

TRAIN_PATH = "data/train.csv"

df = pd.read_csv(TRAIN_PATH)

print("=" * 60)
print("TITANIC SURVIVAL PREDICTION - DATA ANALYSIS")
print("=" * 60)

print("\nShape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types and information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe(include="all"))

print("\nTarget distribution:")
print(df["Survived"].value_counts())
