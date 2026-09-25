# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 03_bagging.py
#
# Purpose:
#   Fit and evaluate a bagging classification model for the
#   NObeyesdad target.
#
#   This script:
#     1. Loads the training data.
#     2. Separates predictors and target.
#     3. Identifies numeric and categorical predictors.
#     4. Builds a preprocessing pipeline.
#     5. Fits a bagging classifier.
#     6. Evaluates the model on a held-out validation set.
#     7. Saves durable results for Report.Rmd.
#
#   The script does NOT depend on BART.
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
RESULT_DIR = OUTPUT_DIR / "results"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGET = "NObeyesdad"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

train = pd.read_csv(
    DATA_DIR / "train.csv"
)


# ------------------------------------------------------------
# Separate predictors and target
# ------------------------------------------------------------

X = train.drop(
    columns=[TARGET]
)

y = train[TARGET]


# ------------------------------------------------------------
# Remove identifier from predictors
# ------------------------------------------------------------

if "id" in X.columns:
    X = X.drop(
        columns=["id"]
    )


# ------------------------------------------------------------
# Predictor definitions
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
# Console: predictor definitions
# ------------------------------------------------------------

print("Categorical predictors:")
print(categorical_predictors)

print("\nNumeric predictors:")
print(numeric_predictors)


# ------------------------------------------------------------
# Train/validation split
# ------------------------------------------------------------

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_predictors
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_predictors
        )
    ]
)


# ------------------------------------------------------------
# Bagging model
# ------------------------------------------------------------

base_tree = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)

bagging_model = BaggingClassifier(
    estimator=base_tree,
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=1
)


# ------------------------------------------------------------
# Complete modeling pipeline
# ------------------------------------------------------------

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            bagging_model
        )
    ]
)


# ------------------------------------------------------------
# Fit model
# ------------------------------------------------------------

print("\nFitting bagging model...")

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

y_pred = model.predict(
    X_valid
)


# ------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_valid,
    y_pred
)

macro_f1 = f1_score(
    y_valid,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_valid,
    y_pred,
    average="weighted"
)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

report_dict = classification_report(
    y_valid,
    y_pred,
    output_dict=True
)

classification_df = (
    pd.DataFrame(report_dict)
    .T
    .reset_index()
    .rename(
        columns={
            "index": "Class"
        }
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

class_labels = sorted(
    y.unique()
)

confusion_df = pd.DataFrame(
    confusion_matrix(
        y_valid,
        y_pred,
        labels=class_labels
    ),
    index=class_labels,
    columns=class_labels
)

confusion_df.index.name = "Actual"


# ------------------------------------------------------------
# Model metrics table
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    {
        "Model": [
            "Bagging"
        ],
        "Accuracy": [
            accuracy
        ],
        "Macro_F1": [
            macro_f1
        ],
        "Weighted_F1": [
            weighted_f1
        ],
        "Validation_Observations": [
            len(y_valid)
        ],
        "Training_Observations": [
            len(y_train)
        ]
    }
)


# ------------------------------------------------------------
# Save durable artifacts
# ------------------------------------------------------------

metrics_df.to_csv(
    RESULT_DIR / "bagging_metrics.csv",
    index=False
)

classification_df.to_csv(
    RESULT_DIR / "bagging_classification_report.csv",
    index=False
)

confusion_df.to_csv(
    RESULT_DIR / "bagging_confusion_matrix.csv"
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("\n==============================")
print("BAGGING")
print("==============================")

print(
    f"Accuracy:     {accuracy:.4f}"
)

print(
    f"Macro F1:     {macro_f1:.4f}"
)

print(
    f"Weighted F1:  {weighted_f1:.4f}"
)

print("\nClassification Report:")

print(
    classification_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Artifact confirmation
# ------------------------------------------------------------

print("\nRESULT ARTIFACTS SAVED")
print("=" * 60)

for artifact in sorted(
    RESULT_DIR.glob("bagging_*.csv")
):
    print(
        artifact.name
    )

print("\n03_bagging.py completed successfully.")
