import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("outputs/plots", exist_ok=True)

df = pd.read_csv("data/train.csv")

print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

# Target distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Survived")
plt.title("Survival Distribution")
plt.tight_layout()
plt.savefig("outputs/plots/survival_distribution.png", dpi=200)
plt.close()

# Gender
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.tight_layout()
plt.savefig("outputs/plots/survival_by_gender.png", dpi=200)
plt.close()

# Passenger class
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.tight_layout()
plt.savefig("outputs/plots/survival_by_class.png", dpi=200)
plt.close()

# Age
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Age", hue="Survived", kde=True)
plt.title("Age Distribution by Survival")
plt.tight_layout()
plt.savefig("outputs/plots/age_distribution.png", dpi=200)
plt.close()

# Fare
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="Fare", kde=True)
plt.title("Fare Distribution")
plt.tight_layout()
plt.savefig("outputs/plots/fare_distribution.png", dpi=200)
plt.close()

# Correlation
plt.figure(figsize=(10, 7))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/plots/correlation_heatmap.png", dpi=200)
plt.close()

# Outliers
for column in ["Age", "Fare", "SibSp", "Parch"]:
    plt.figure(figsize=(7, 4))
    sns.boxplot(x=df[column])
    plt.title(f"Boxplot - {column}")
    plt.tight_layout()
    plt.savefig(
        f"outputs/plots/boxplot_{column.lower()}.png",
        dpi=200
    )
    plt.close()

print("\nEDA plots saved in outputs/plots/")
