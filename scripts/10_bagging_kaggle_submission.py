# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 10_bagging_kaggle_submission.py
#
# Purpose:
#   Fit and evaluate a bagged decision-tree classification
#   model for the NObeyesdad target.
#
#   This script:
#     1. Loads the training and test data.
#     2. Separates predictors and target.
#     3. Removes the identifier variable.
#     4. Identifies numeric and categorical predictors.
#     5. Builds a preprocessing pipeline.
#     6. Fits a bagged decision-tree classifier.
#     7. Evaluates the model on a held-out validation set.
#     8. Fits a final model using all labeled training data.
#     9. Generates Kaggle test predictions.
#    10. Saves durable results for Report.Rmd.
#
#   Bagging reduces variance by fitting multiple decision
#   trees on bootstrap samples and aggregating their predictions.
#
#   BART is NOT called by this script.
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


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

RESULT_DIR = OUTPUT_DIR / "results"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "NObeyesdad"

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_ESTIMATORS = 300


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BAGGING")
print("=" * 70)

print("\nLoading data...")

train = pd.read_csv(
    DATA_DIR / "train.csv"
)

test = pd.read_csv(
    DATA_DIR / "test.csv"
)

print(
    f"Training observations: {len(train):,}"
)

print(
    f"Training variables:    {len(train.columns):,}"
)

print(
    f"Test observations:     {len(test):,}"
)

print(
    f"Test variables:        {len(test.columns):,}"
)


# ============================================================
# VALIDATE TARGET
# ============================================================

if TARGET not in train.columns:

    raise ValueError(
        f"Target variable '{TARGET}' was not found "
        "in train.csv."
    )


# ============================================================
# SEPARATE PREDICTORS AND TARGET
# ============================================================

X = train.drop(
    columns=[TARGET]
)

y = train[TARGET].astype(str)


# ============================================================
# REMOVE IDENTIFIER
# ============================================================

if "id" in X.columns:

    X = X.drop(
        columns=["id"]
    )


if "id" in test.columns:

    test_ids = test["id"].copy()

    X_test_kaggle = test.drop(
        columns=["id"]
    )

else:

    raise ValueError(
        "The 'id' column was not found in test.csv."
    )


# ============================================================
# PREDICTOR DEFINITIONS
# ============================================================

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


expected_predictors = (
    numeric_predictors
    + categorical_predictors
)


# ============================================================
# VALIDATE PREDICTORS
# ============================================================

missing_training_predictors = [
    column
    for column in expected_predictors
    if column not in X.columns
]

if missing_training_predictors:

    raise ValueError(
        "The following expected predictors are missing "
        "from the training data: "
        + str(missing_training_predictors)
    )


missing_test_predictors = [
    column
    for column in expected_predictors
    if column not in X_test_kaggle.columns
]

if missing_test_predictors:

    raise ValueError(
        "The following expected predictors are missing "
        "from the test data: "
        + str(missing_test_predictors)
    )


X = X[
    expected_predictors
]

X_test_kaggle = X_test_kaggle[
    expected_predictors
]


# ============================================================
# CONSOLE: TARGET CLASSES
# ============================================================

class_labels = sorted(
    y.unique()
)

print("\nTarget classes:")

for class_label in class_labels:

    print(
        f"  {class_label}"
    )


# ============================================================
# CONSOLE: PREDICTOR DEFINITIONS
# ============================================================

print("\nNumeric predictors:")

for predictor in numeric_predictors:

    print(
        f"  {predictor}"
    )


print("\nCategorical predictors:")

for predictor in categorical_predictors:

    print(
        f"  {predictor}"
    )


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

print(
    "\nCreating stratified train/validation split..."
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"Training observations:   {len(y_train):,}"
)

print(
    f"Validation observations: {len(y_valid):,}"
)

print(
    f"Validation proportion:   {TEST_SIZE:.0%}"
)

print(
    f"Random state:            {RANDOM_STATE}"
)


# ============================================================
# PREPROCESSING
# ============================================================

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


# ============================================================
# BAGGING MODEL
# ============================================================
#
# Bagging fits multiple decision trees using bootstrap
# samples of the training data and aggregates their
# predictions.
#
# The base decision trees are intentionally left unpruned.
# Bagging controls variance through aggregation across
# bootstrap samples.
#
# n_estimators = 300 provides a sufficiently large ensemble
# while remaining practical for this data set.
# ============================================================

base_tree = DecisionTreeClassifier(
    random_state=RANDOM_STATE
)


bagging_model = BaggingClassifier(
    estimator=base_tree,
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=1
)


# ============================================================
# COMPLETE MODELING PIPELINE
# ============================================================

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


# ============================================================
# FIT VALIDATION MODEL
# ============================================================

print(
    "\nFitting bagged decision-tree model..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Model fitting completed."
)


# ============================================================
# VALIDATION PREDICTIONS
# ============================================================

print(
    "\nGenerating validation predictions..."
)

y_pred = model.predict(
    X_valid
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_valid,
    y_pred
)

macro_f1 = f1_score(
    y_valid,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_valid,
    y_pred,
    average="weighted",
    zero_division=0
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_valid,
    y_pred,
    labels=class_labels,
    target_names=class_labels,
    output_dict=True,
    zero_division=0
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


# ============================================================
# CONFUSION MATRIX
# ============================================================

confusion = confusion_matrix(
    y_valid,
    y_pred,
    labels=class_labels
)


confusion_df = pd.DataFrame(
    confusion,
    index=class_labels,
    columns=class_labels
)


confusion_df.index.name = "Actual"

confusion_df.columns.name = "Predicted"


# ============================================================
# MODEL METRICS TABLE
# ============================================================

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
        ],
        "Number_of_Estimators": [
            N_ESTIMATORS
        ],
        "Random_State": [
            RANDOM_STATE
        ],
        "Validation_Proportion": [
            TEST_SIZE
        ]
    }
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================
#
# Each tree in the bagging ensemble has its own feature
# importance values. The final importance below is the
# average importance across all fitted trees.
#
# These values describe how the fitted ensemble used the
# predictors; they should not be interpreted as causal effects.
# ============================================================

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


tree_importances = []

for estimator in classifier.estimators_:

    tree_importances.append(
        estimator.feature_importances_
    )


importance_df = pd.DataFrame(
    tree_importances,
    columns=feature_names
)


importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": importance_df.mean(
            axis=0
        ).values
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


# ============================================================
# SAVE VALIDATION ARTIFACTS
# ============================================================

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


importance_df.to_csv(
    RESULT_DIR / "bagging_feature_importance.csv",
    index=False
)


# ============================================================
# CONFIGURATION ARTIFACT
# ============================================================

configuration_df = pd.DataFrame(
    {
        "Parameter": [
            "Model",
            "Base Estimator",
            "Number of Estimators",
            "Random State",
            "Validation Proportion",
            "Training Observations",
            "Validation Observations",
            "Predictor Encoding",
            "ID Excluded",
            "Bootstrap Sampling"
        ],
        "Value": [
            "BaggingClassifier",
            "DecisionTreeClassifier",
            N_ESTIMATORS,
            RANDOM_STATE,
            TEST_SIZE,
            len(y_train),
            len(y_valid),
            "One-hot encoding for categorical predictors",
            "Yes",
            "Yes"
        ]
    }
)


configuration_df.to_csv(
    RESULT_DIR / "bagging_configuration.csv",
    index=False
)


# ============================================================
# CONSOLE: VALIDATION RESULTS
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "BAGGING VALIDATION RESULTS"
)

print(
    "=" * 60
)


print(
    f"Accuracy:     {accuracy:.4f}"
)

print(
    f"Macro F1:     {macro_f1:.4f}"
)

print(
    f"Weighted F1:  {weighted_f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)

print(
    classification_df.to_string(
        index=False
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)

print(
    confusion_df.to_string()
)


# ============================================================
# TOP FEATURES
# ============================================================

print(
    "\nTOP 10 BAGGING FEATURES"
)

print(
    "=" * 60
)

print(
    importance_df.head(10).to_string(
        index=False
    )
)


# ============================================================
# FINAL MODEL
# ============================================================
#
# After validation performance has been calculated, the final
# bagged model is fitted using all labeled training data.
#
# This final model is used only to generate the Kaggle
# competition predictions.
# ============================================================

print(
    "\nFitting final bagging model using all training data..."
)


final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            BaggingClassifier(
                estimator=DecisionTreeClassifier(
                    random_state=RANDOM_STATE
                ),
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=1
            )
        )
    ]
)


final_model.fit(
    X,
    y
)


print(
    "Final model fitting completed."
)


# ============================================================
# KAGGLE TEST PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions for Kaggle test data..."
)

test_predictions = final_model.predict(
    X_test_kaggle
)


# ============================================================
# KAGGLE SUBMISSION
# ============================================================
#
# The submission contains exactly two columns:
#
#     id
#     NObeyesdad
#
# The ID values come directly from test.csv and are not
# modified by the model.
# ============================================================

submission = pd.DataFrame(
    {
        "id": test_ids,
        TARGET: test_predictions
    }
)


submission_path = (
    SUBMISSION_DIR
    / "bagging_submission.csv"
)


submission.to_csv(
    submission_path,
    index=False
)


# ============================================================
# VALIDATE SUBMISSION STRUCTURE
# ============================================================

expected_submission_columns = [
    "id",
    TARGET
]


if submission.columns.tolist() != expected_submission_columns:

    raise ValueError(
        "Submission columns do not match the required "
        "format: ['id', 'NObeyesdad']"
    )


if len(submission) != len(test):

    raise ValueError(
        "Submission row count does not match test.csv."
    )


if submission["id"].tolist() != test_ids.tolist():

    raise ValueError(
        "Submission IDs do not match the test.csv IDs."
    )


if submission[TARGET].isna().any():

    raise ValueError(
        "Submission contains missing NObeyesdad predictions."
    )


# ============================================================
# SUBMISSION SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "KAGGLE SUBMISSION"
)

print(
    "=" * 60
)

print(
    f"Rows:    {len(submission):,}"
)

print(
    f"Columns: {submission.columns.tolist()}"
)

print(
    f"File:    {submission_path}"
)


print(
    "\nPrediction distribution:"
)

print(
    submission[TARGET]
    .value_counts()
    .sort_index()
)


# ============================================================
# FINAL ARTIFACT CONFIRMATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "RESULT ARTIFACTS SAVED"
)

print(
    "=" * 70
)


for artifact in sorted(
    RESULT_DIR.glob(
        "bagging_*.csv"
    )
):

    print(
        artifact.name
    )


print(
    "bagging_submission.csv"
)

print(
    "\n10_bagging_kaggle_submission.py completed successfully."
)
