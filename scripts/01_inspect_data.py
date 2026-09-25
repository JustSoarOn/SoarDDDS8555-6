# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 01_inspect_data.py
#
# Purpose:
#   Inspect the Kaggle training/test data and save durable
#   report-ready artifacts for SoarDDDS8555-6-Report.Rmd.
# ============================================================

from pathlib import Path
import pandas as pd


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
TABLE_DIR = OUTPUT_DIR / "tables"

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGET = "NObeyesdad"


# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

train = pd.read_csv(
    DATA_DIR / "train.csv"
)

test = pd.read_csv(
    DATA_DIR / "test.csv"
)

sample_submission = pd.read_csv(
    DATA_DIR / "sample_submission.csv"
)


# ------------------------------------------------------------
# Basic dimensions
# ------------------------------------------------------------

data_dimensions = pd.DataFrame(
    {
        "Dataset": [
            "Training",
            "Test",
            "Sample Submission"
        ],
        "Rows": [
            train.shape[0],
            test.shape[0],
            sample_submission.shape[0]
        ],
        "Columns": [
            train.shape[1],
            test.shape[1],
            sample_submission.shape[1]
        ]
    }
)

data_dimensions.to_csv(
    TABLE_DIR / "data_dimensions.csv",
    index=False
)


# ------------------------------------------------------------
# Console: Basic dimensions
# ------------------------------------------------------------

print("TRAINING DATA")
print("=" * 60)
print(f"Rows:    {train.shape[0]:,}")
print(f"Columns: {train.shape[1]:,}")

print("\nTEST DATA")
print("=" * 60)
print(f"Rows:    {test.shape[0]:,}")
print(f"Columns: {test.shape[1]:,}")

print("\nSAMPLE SUBMISSION")
print("=" * 60)
print(f"Rows:    {sample_submission.shape[0]:,}")
print(f"Columns: {sample_submission.shape[1]:,}")


# ------------------------------------------------------------
# Variable names and data types
# ------------------------------------------------------------

data_types = pd.DataFrame(
    {
        "Variable": train.columns,
        "Data_Type": [
            str(dtype)
            for dtype in train.dtypes
        ]
    }
)

data_types.to_csv(
    TABLE_DIR / "data_types.csv",
    index=False
)


# ------------------------------------------------------------
# Console: Training variables
# ------------------------------------------------------------

print("\nTRAINING VARIABLES")
print("=" * 60)

for column in train.columns:
    print(
        f"{column}: {train[column].dtype}"
    )


# ------------------------------------------------------------
# Target distribution
# ------------------------------------------------------------

target_counts = (
    train[TARGET]
    .value_counts()
    .rename_axis("NObeyesdad")
    .reset_index(name="Count")
)

target_counts["Proportion"] = (
    target_counts["Count"]
    / target_counts["Count"].sum()
)

target_counts["Percentage"] = (
    target_counts["Proportion"] * 100
)

target_counts.to_csv(
    TABLE_DIR / "target_distribution.csv",
    index=False
)


# ------------------------------------------------------------
# Console: Target variable
# ------------------------------------------------------------

print("\nTARGET VARIABLE")
print("=" * 60)
print(TARGET)

print("\nTarget classes:")
print(
    train[TARGET]
    .value_counts()
)

print("\nTarget proportions:")
print(
    train[TARGET]
    .value_counts(
        normalize=True
    ).round(4)
)


# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

missing_values = pd.DataFrame(
    {
        "Variable": train.columns,
        "Missing_Count": [
            int(train[column].isna().sum())
            for column in train.columns
        ]
    }
)

missing_values["Missing_Percentage"] = (
    missing_values["Missing_Count"]
    / len(train)
    * 100
)

missing_values.to_csv(
    TABLE_DIR / "missing_values.csv",
    index=False
)


# ------------------------------------------------------------
# Console: Missing values
# ------------------------------------------------------------

print("\nMISSING VALUES")
print("=" * 60)
print(
    train.isna().sum()
)


# ------------------------------------------------------------
# Duplicate observations
# ------------------------------------------------------------

duplicate_count = int(
    train.duplicated().sum()
)

duplicate_summary = pd.DataFrame(
    {
        "Measure": [
            "Duplicate training rows"
        ],
        "Count": [
            duplicate_count
        ]
    }
)

duplicate_summary.to_csv(
    TABLE_DIR / "duplicate_summary.csv",
    index=False
)


# ------------------------------------------------------------
# Console: Duplicates
# ------------------------------------------------------------

print("\nDUPLICATES")
print("=" * 60)
print(
    "Duplicate training rows:",
    duplicate_count
)


# ------------------------------------------------------------
# First five observations
# ------------------------------------------------------------

first_five = train.head(5)

first_five.to_csv(
    TABLE_DIR / "first_five_training_observations.csv",
    index=False
)


# ------------------------------------------------------------
# Console: First observations
# ------------------------------------------------------------

print("\nFIRST FIVE TRAINING OBSERVATIONS")
print("=" * 60)
print(
    first_five
)


# ------------------------------------------------------------
# Test and submission columns
# ------------------------------------------------------------

test_columns = pd.DataFrame(
    {
        "Variable": test.columns
    }
)

submission_columns = pd.DataFrame(
    {
        "Variable": sample_submission.columns
    }
)

test_columns.to_csv(
    TABLE_DIR / "test_variables.csv",
    index=False
)

submission_columns.to_csv(
    TABLE_DIR / "submission_variables.csv",
    index=False
)


# ------------------------------------------------------------
# Console: Test/submission columns
# ------------------------------------------------------------

print("\nTEST VARIABLES")
print("=" * 60)
print(
    test.columns.tolist()
)

print("\nSUBMISSION VARIABLES")
print("=" * 60)
print(
    sample_submission.columns.tolist()
)


# ------------------------------------------------------------
# Final artifact confirmation
# ------------------------------------------------------------

print("\nREPORT ARTIFACTS SAVED")
print("=" * 60)

for artifact in sorted(
    TABLE_DIR.glob("*.csv")
):
    print(
        artifact.name
    )

print("\n01_inspect_data.py completed successfully.")
