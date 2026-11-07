import os

import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def header(text):
    print("\n" + "=" * 90 + "\n" + text + "\n" + "=" * 90)


# ------------------------------------------------------------------------------------------
header("Q1. PRACTICAL IMPLEMENTATION & PREPROCESSING")
# ------------------------------------------------------------------------------------------

# 1.1 Load dataset
df = pd.read_csv(os.path.join(BASE, "framingham.csv"))
print("Shape:", df.shape)
print("\nColumns and data types:\n", df.dtypes, sep="")

# 1.2 Missing values
TARGET = "TenYearCHD"
MODE_COLS = ["education", "BPMeds"]
MEDIAN_COLS = ["cigsPerDay", "totChol", "BMI", "heartRate", "glucose"]

missing = df.isna().sum()
print("\nMissing values:\n", missing[missing > 0], sep="")
print("\nImputation: median for", MEDIAN_COLS, "| mode for", MODE_COLS)
smokers_missing = df.loc[df["cigsPerDay"].isna(), "currentSmoker"]
print(f"All {len(smokers_missing)} missing cigsPerDay rows are smokers = {(smokers_missing == 1).all()}, "
      f"so they get the smokers' median ({df.loc[df['currentSmoker'] == 1, 'cigsPerDay'].median():.0f}) "
      f"instead of the overall median ({df['cigsPerDay'].median():.0f}).")


class CHDImputer(BaseEstimator, TransformerMixin):
    """Median/mode imputation using statistics learned from the training data only."""

    def fit(self, X, y=None):
        self.medians_ = X[MEDIAN_COLS].median()
        self.modes_ = X[MODE_COLS].mode().iloc[0]
        self.smoker_cigs_ = X.loc[X["currentSmoker"] == 1, "cigsPerDay"].median()
        return self

    def transform(self, X):
        X = X.copy()
        smokers = X["currentSmoker"] == 1
        X.loc[smokers, "cigsPerDay"] = X.loc[smokers, "cigsPerDay"].fillna(self.smoker_cigs_)
        X[MODE_COLS] = X[MODE_COLS].fillna(self.modes_)
        return X.fillna(self.medians_)


# 1.3 Duplicates
print("\nDuplicate records:", df.duplicated().sum())
df = df.drop_duplicates()
print("Shape after removing duplicates:", df.shape)

# 1.4 Encoding and feature/target separation
print("\nEncoding: every column is already numeric. Binary columns are 0/1 and education is an")
print("ordered 1-4 scale, so no further encoding is required.")
for col in ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes", "education"]:
    print(f"  {col:16s} values = {sorted(df[col].dropna().unique().tolist())}")

X = df.drop(columns=TARGET)
y = df[TARGET]

# 1.5 Stratified 80:20 split. Imputer, scaler and SMOTE sit inside pipelines, so they only ever
# learn from training data.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
print(f"\nTrain: {X_train.shape} ({y_train.mean():.1%} CHD) | Test: {X_test.shape} ({y_test.mean():.1%} CHD)")

X_train_imp = CHDImputer().fit(X_train).transform(X_train)
scaler = StandardScaler().fit(X_train_imp)
X_train_scaled = pd.DataFrame(scaler.transform(X_train_imp), columns=X.columns, index=X_train.index)
print("Scaled training data: mean = {:.3f}, std = {:.3f}".format(
    X_train_scaled.values.mean(), X_train_scaled.values.std()))


# ------------------------------------------------------------------------------------------
header("Q2. EXPLORATORY DATA ANALYSIS")
# ------------------------------------------------------------------------------------------

# 2.1 Descriptive statistics and target distribution
print(df.describe().T.round(2))
counts = df[TARGET].value_counts()
baseline_acc = counts.max() / counts.sum()
print(f"\nTarget: No CHD = {counts[0]} ({counts[0] / len(df):.1%}), CHD = {counts[1]} ({counts[1] / len(df):.1%})")
print(f"Always predicting 'No CHD' already scores {baseline_acc:.1%} accuracy.")

eda = df.assign(CHD=df[TARGET].map({0: "No CHD", 1: "CHD"}))
CONTINUOUS = ["age", "cigsPerDay", "totChol", "sysBP", "diaBP", "BMI", "heartRate", "glucose"]
CATEGORICAL = ["male", "education", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp", "diabetes"]
eda[CATEGORICAL] = eda[CATEGORICAL].astype("Int64")
