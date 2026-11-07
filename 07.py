import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "07_outputs")
os.makedirs(OUT, exist_ok=True)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CLASS_COLORS = {"No CHD": BLUE, "CHD": ORANGE}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "figure.titlesize": 14,
    "figure.titleweight": "bold", "legend.frameon": False, "savefig.dpi": 150,
})


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close()


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

# 2.2 Plots
fig, ax = plt.subplots(figsize=(5, 4))
bars = ax.bar(["No CHD", "CHD"], [counts[0], counts[1]], color=[BLUE, ORANGE], width=0.6)
ax.bar_label(bars, labels=[f"{counts[0]} ({counts[0] / len(df):.1%})", f"{counts[1]} ({counts[1] / len(df):.1%})"], padding=3)
ax.set_title("Only 15% of patients developed CHD")
ax.set_ylabel("Patients")
ax.grid(axis="x", visible=False)
ax.margins(y=0.12)
save("01_target_distribution.png")

fig, ax = plt.subplots(figsize=(11, 9))
diverging = LinearSegmentedColormap.from_list("blue_red", [BLUE, "#f0efec", "#e34948"])
sns.heatmap(df.corr(), cmap=diverging, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 8},
            linewidths=1, linecolor=SURFACE, cbar_kws={"label": "Correlation", "shrink": 0.8}, ax=ax)
ax.set_title("No single feature correlates strongly with TenYearCHD (age is highest at 0.23)")
save("02_correlation_heatmap.png")

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.ravel(), CONTINUOUS):
    ax.hist(df[col].dropna(), bins=30, color=BLUE, edgecolor=SURFACE, linewidth=0.6)
    ax.set_title(col)
    ax.grid(axis="x", visible=False)
fig.suptitle("Distribution of continuous features")
save("03_histograms.png")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.ravel(), CONTINUOUS):
    sns.boxplot(data=eda, x="CHD", y=col, hue="CHD", palette=CLASS_COLORS, order=["No CHD", "CHD"],
                legend=False, width=0.55, linewidth=1, fliersize=2, saturation=1, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(axis="x", visible=False)
fig.suptitle("Continuous features by 10-year CHD outcome")
save("04_boxplots.png")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, col in zip(axes.ravel(), CATEGORICAL):
    sns.countplot(data=eda, x=col, hue="CHD", palette=CLASS_COLORS, hue_order=["No CHD", "CHD"],
                  legend=False, gap=0.08, saturation=1, ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Patients")
    ax.grid(axis="x", visible=False)
axes[1, 3].axis("off")
axes[1, 3].legend(handles=[Patch(color=c, label=k) for k, c in CLASS_COLORS.items()], loc="center", fontsize=12)
fig.suptitle("Categorical features by 10-year CHD outcome")
save("05_countplots.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (a, b) in zip(axes, [("age", "sysBP"), ("totChol", "sysBP"), ("BMI", "glucose")]):
    for label in ["No CHD", "CHD"]:
        part = eda[eda["CHD"] == label]
        ax.scatter(part[a], part[b], s=16, alpha=0.45, color=CLASS_COLORS[label], edgecolor=SURFACE,
                   linewidth=0.3, label=label)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title(f"{a} vs {b}")
axes[0].legend(loc="upper left")
fig.suptitle("The two classes overlap heavily, which limits any classifier")
save("06_scatterplots.png")

# 2.3 Key relationships
bands = {
    "age": ([30, 40, 50, 60, 70], ["31-40", "41-50", "51-60", "61-70"]),
    "sysBP": ([0, 120, 140, 160, 300], ["<=120", "121-140", "141-160", ">160"]),
    "totChol": ([0, 200, 240, 800], ["<=200", "201-240", ">240"]),
    "glucose": ([0, 100, 126, 500], ["<=100", "101-126", ">126"]),
    "cigsPerDay": ([-1, 0, 10, 20, 100], ["0", "1-10", "11-20", ">20"]),
}
overall_rate = df[TARGET].mean()
rates = {}
fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)
for ax, (col, (edges, labels)) in zip(axes, bands.items()):
    group = df.groupby(pd.cut(df[col], edges, labels=labels), observed=True)[TARGET].agg(["mean", "size"])
    rates[col] = group
    print(f"\nCHD rate by {col}:\n", group.round(3), sep="")
    bars = ax.bar(group.index.astype(str), group["mean"] * 100, color=BLUE, width=0.65)
    ax.bar_label(bars, fmt="%.0f%%", padding=2, fontsize=9, zorder=5,
                 bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8})
    ax.axhline(overall_rate * 100, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title(col)
    ax.grid(axis="x", visible=False)
axes[0].set_ylabel("10-year CHD rate (%)")
axes[0].text(-0.45, overall_rate * 100 + 1, f"overall {overall_rate:.0%}", ha="left", color=INK2, fontsize=9)
fig.suptitle("CHD risk climbs with age, blood pressure, glucose and heavy smoking")
save("07_chd_rate_by_risk_factor.png")

smoke = df.groupby("currentSmoker")[TARGET].mean()
hyp = df.groupby("prevalentHyp")[TARGET].mean()
print("\nActionable insights (associations, not proof of cause):")
print(f"1. Age: CHD rate rises from {rates['age']['mean'].iloc[0]:.1%} (31-40) to {rates['age']['mean'].iloc[-1]:.1%} (61-70). "
      "Prioritise routine screening from age 50.")
print(f"2. Blood pressure: {rates['sysBP']['mean'].iloc[-1]:.1%} CHD above 160 mmHg vs {rates['sysBP']['mean'].iloc[0]:.1%} at or below 120; "
      f"hypertensive patients {hyp[1]:.1%} vs {hyp[0]:.1%}. BP control is the strongest modifiable target here.")
print(f"3. Glucose: {rates['glucose']['mean'].iloc[-1]:.1%} CHD above 126 mg/dL vs {rates['glucose']['mean'].iloc[0]:.1%} at or below 100 "
      f"(only {rates['glucose']['size'].iloc[-1]} patients). Flag high fasting glucose for cardiac follow-up.")
print(f"4. Smoking: smokers {smoke[1]:.1%} vs non-smokers {smoke[0]:.1%} is a small gap, but over 20 cigarettes/day reaches "
      f"{rates['cigsPerDay']['mean'].iloc[-1]:.1%}. Record the amount smoked, not just yes/no.")
print(f"5. Cholesterol: {rates['totChol']['mean'].iloc[-1]:.1%} CHD above 240 mg/dL vs {rates['totChol']['mean'].iloc[0]:.1%} at or below 200.")
