# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 05_random_forest.py
#
# Purpose:
#   Fit and evaluate a Random Forest classification model for
#   the NObeyesdad target.
#
#   This script:
#     1. Loads the training data.
#     2. Separates predictors and target.
#     3. Removes the identifier variable.
#     4. Identifies numeric and categorical predictors.
#     5. Builds a preprocessing pipeline.
#     6. Fits a Random Forest classifier.
#     7. Evaluates the model on a held-out validation set.
#     8. Saves durable results for Report.Rmd.
#
#   BART is NOT called by this script.
# ============================================================

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


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
# Random Forest model
# ------------------------------------------------------------

random_forest_model = RandomForestClassifier(
    n_estimators=300,
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
            random_forest_model
        )
    ]
)


# ------------------------------------------------------------
# Fit model
# ------------------------------------------------------------

print("\nFitting random forest model...")

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
            "Random Forest"
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
# Feature importance
# ------------------------------------------------------------

classifier = model.named_steps[
    "classifier"
]

fitted_preprocessor = model.named_steps[
    "preprocessor"
]

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": classifier.feature_importances_
    }
)

importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Save durable artifacts
# ------------------------------------------------------------

metrics_df.to_csv(
    RESULT_DIR / "random_forest_metrics.csv",
    index=False
)

classification_df.to_csv(
    RESULT_DIR / "random_forest_classification_report.csv",
    index=False
)

confusion_df.to_csv(
    RESULT_DIR / "random_forest_confusion_matrix.csv"
)

importance_df.to_csv(
    RESULT_DIR / "random_forest_feature_importance.csv",
    index=False
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print("\n==============================")
print("RANDOM FOREST")
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
# Top features
# ------------------------------------------------------------

print("\nTOP 10 FEATURES")
print("=" * 60)

print(
    importance_df.head(10).to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Artifact confirmation
# ------------------------------------------------------------

print("\nRESULT ARTIFACTS SAVED")
print("=" * 60)

for artifact in sorted(
    RESULT_DIR.glob("random_forest_*.csv")
):
    print(
        artifact.name
    )

print("\n05_random_forest.py completed successfully.")

