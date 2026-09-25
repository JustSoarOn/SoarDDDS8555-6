# ============================================================
# Assignment 6: Build and Evaluate Tree Models
# 02_eda.py
#
# Purpose:
#   Exploratory data analysis for the obesity classification
#   problem.
#
#        02_eda.py
#
#   This script:
#     1. Loads the training data.
#     2. Creates numerical and categorical summaries.
#     3. Examines the target distribution.
#     4. Examines relationships between numeric predictors.
#     5. Creates report-ready figures.
#     6. Saves all durable artifacts for Report.Rmd.
#
#   No predictive model is fitted in this script.
# ============================================================

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGET = "NObeyesdad"

sns.set_theme(
    style="whitegrid",
    context="notebook"
)


# ------------------------------------------------------------
# Load training data
# ------------------------------------------------------------

train = pd.read_csv(
    DATA_DIR / "train.csv"
)


# ------------------------------------------------------------
# Identify predictor types
# ------------------------------------------------------------

numeric_predictors = [
    "Age",
    "Height",
    "Weight",
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]

categorical_predictors = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]


# ------------------------------------------------------------
# Numeric summary statistics
# ------------------------------------------------------------

numeric_summary = (
    train[numeric_predictors]
    .describe()
    .T
    .reset_index()
    .rename(
        columns={
            "index": "Variable"
        }
    )
)

numeric_summary.to_csv(
    TABLE_DIR / "numeric_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Categorical summary statistics
# ------------------------------------------------------------

categorical_rows = []

for variable in categorical_predictors:

    counts = (
        train[variable]
        .value_counts(dropna=False)
    )

    proportions = (
        train[variable]
        .value_counts(
            normalize=True,
            dropna=False
        )
    )

    for category in counts.index:

        categorical_rows.append(
            {
                "Variable": variable,
                "Category": category,
                "Count": int(counts.loc[category]),
                "Proportion": float(
                    proportions.loc[category]
                ),
                "Percentage": float(
                    proportions.loc[category] * 100
                )
            }
        )

categorical_summary = pd.DataFrame(
    categorical_rows
)

categorical_summary.to_csv(
    TABLE_DIR / "categorical_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------

target_distribution = (
    train[TARGET]
    .value_counts()
    .rename_axis(TARGET)
    .reset_index(name="Count")
)

target_distribution["Proportion"] = (
    target_distribution["Count"]
    / target_distribution["Count"].sum()
)

target_distribution["Percentage"] = (
    target_distribution["Proportion"] * 100
)

target_distribution.to_csv(
    TABLE_DIR / "eda_target_distribution.csv",
    index=False
)


# ------------------------------------------------------------
# Correlation matrix
# ------------------------------------------------------------

correlation_matrix = (
    train[numeric_predictors]
    .corr()
)

correlation_matrix.to_csv(
    TABLE_DIR / "numeric_correlation_matrix.csv"
)


# ------------------------------------------------------------
# Numeric predictors by target class
# ------------------------------------------------------------

target_numeric_summary = (
    train
    .groupby(TARGET)[numeric_predictors]
    .agg(
        [
            "mean",
            "median",
            "std"
        ]
    )
)

target_numeric_summary.to_csv(
    TABLE_DIR / "numeric_summary_by_target.csv"
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nNumeric predictors:")
print(numeric_predictors)

print("\nCategorical predictors:")
print(categorical_predictors)

print("\nNUMERIC SUMMARY")
print("=" * 60)
print(numeric_summary.round(3).to_string(index=False))

print("\nTARGET DISTRIBUTION")
print("=" * 60)
print(target_distribution.round(4).to_string(index=False))

print("\nNUMERIC CORRELATION MATRIX")
print("=" * 60)
print(correlation_matrix.round(3).to_string())

print("\nEDA ARTIFACTS BEING CREATED")
print("=" * 60)


# ------------------------------------------------------------
# Figure 1: Target distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 6)
)

target_plot = (
    train[TARGET]
    .value_counts()
    .sort_values(
        ascending=True
    )
)

sns.barplot(
    x=target_plot.values,
    y=target_plot.index,
    color="#4472C4"
)

plt.title(
    "Distribution of Obesity Classification"
)

plt.xlabel(
    "Number of Observations"
)

plt.ylabel(
    "NObeyesdad"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Figure 2: Numeric correlation heatmap
# ------------------------------------------------------------

plt.figure(
    figsize=(10, 8)
)

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title(
    "Correlation Matrix of Numeric Predictors"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "numeric_correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Figure 3: Weight by target class
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

sns.boxplot(
    data=train,
    x=TARGET,
    y="Weight",
    color="#70AD47"
)

plt.title(
    "Weight Distribution by Obesity Classification"
)

plt.xlabel(
    "NObeyesdad"
)

plt.ylabel(
    "Weight"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "weight_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Figure 4: Age by target class
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

sns.boxplot(
    data=train,
    x=TARGET,
    y="Age",
    color="#ED7D31"
)

plt.title(
    "Age Distribution by Obesity Classification"
)

plt.xlabel(
    "NObeyesdad"
)

plt.ylabel(
    "Age"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "age_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Figure 5: Height by target class
# ------------------------------------------------------------

plt.figure(
    figsize=(11, 6)
)

sns.boxplot(
    data=train,
    x=TARGET,
    y="Height",
    color="#A5A5A5"
)

plt.title(
    "Height Distribution by Obesity Classification"
)

plt.xlabel(
    "NObeyesdad"
)

plt.ylabel(
    "Height"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "height_by_target.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# Final artifact confirmation
# ------------------------------------------------------------

print("\nTABLE ARTIFACTS")
print("=" * 60)

for artifact in sorted(
    TABLE_DIR.glob("*.csv")
):
    print(
        artifact.name
    )

print("\nFIGURE ARTIFACTS")
print("=" * 60)

for artifact in sorted(
    FIG_DIR.glob("*.png")
):
    print(
        artifact.name
    )

print("\n02_eda.py completed successfully.")
