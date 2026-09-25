# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 12_boosting_kaggle_submission.py
#
# Purpose:
#   Fit and evaluate a boosted-tree classification model for
#   the NObeyesdad target.
#
#   This script:
#     1. Loads the Kaggle training/test data.
#     2. Separates predictors and target.
#     3. Removes the identifier variable.
#     4. Identifies numeric and categorical predictors.
#     5. Ordinal-encodes categorical predictors.
#     6. Fits a HistGradientBoostingClassifier.
#     7. Evaluates the model on a stratified validation set.
#     8. Calculates permutation feature importance using the
#        ORIGINAL predictor columns.
#     9. Saves classification metrics and diagnostics.
#    10. Fits the final model using all training observations.
#    11. Generates Kaggle test predictions.
#    12. Saves a Kaggle-compatible submission file containing
#        exactly the columns: id,NObeyesdad.
#
#   IMPORTANT FEATURE-IMPORTANCE FIX:
#
#   Permutation importance is calculated one ORIGINAL predictor
#   at a time. For categorical variables, the entire original
#   categorical column is shuffled before preprocessing.
#
#   This avoids treating individual ordinal-encoded category
#   values as separate artificial predictors and ensures that
#   every original predictor receives one valid importance
#   value.
#
#   No predictive model is fitted inside the importance loop.
#   The already-fitted validation model is evaluated after each
#   predictor is permuted.
# ============================================================


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OrdinalEncoder


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

TARGET = "NObeyesdad"

RANDOM_STATE = 42

TEST_SIZE = 0.20

# HistGradientBoosting configuration.

N_ESTIMATORS = 300

LEARNING_RATE = 0.05

MAX_LEAF_NODES = 31

L2_REGULARIZATION = 1.0

EARLY_STOPPING = True

# Number of permutations used for each original predictor.

N_PERMUTATIONS = 5


# ------------------------------------------------------------
# Console header
# ------------------------------------------------------------

print(
    "=" * 70
)

print(
    "BOOSTING"
)

print(
    "=" * 70
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print(
    "\nLoading data..."
)

train = pd.read_csv(
    DATA_DIR / "train.csv"
)

test = pd.read_csv(
    DATA_DIR / "test.csv"
)

sample_submission = pd.read_csv(
    DATA_DIR / "sample_submission.csv"
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


# ------------------------------------------------------------
# Validate target
# ------------------------------------------------------------

if TARGET not in train.columns:

    raise ValueError(
        f"Target variable '{TARGET}' was not found "
        "in train.csv."
    )


# ------------------------------------------------------------
# Target classes
# ------------------------------------------------------------

class_labels = sorted(
    train[TARGET].astype(str).unique()
)


print(
    "\nTarget classes:"
)

for class_label in class_labels:

    print(
        f"  {class_label}"
    )


# ------------------------------------------------------------
# Separate predictors and target
# ------------------------------------------------------------

X = train.drop(
    columns=[TARGET]
).copy()

y = train[TARGET].astype(str)


# ------------------------------------------------------------
# Preserve Kaggle test IDs
# ------------------------------------------------------------

if "id" not in train.columns:

    raise ValueError(
        "The training data does not contain the required "
        "'id' column."
    )

if "id" not in test.columns:

    raise ValueError(
        "The test data does not contain the required "
        "'id' column."
    )


test_ids = test["id"].copy()


# ------------------------------------------------------------
# Remove identifier from predictors
# ------------------------------------------------------------

X = X.drop(
    columns=["id"]
)

X_test_kaggle = test.drop(
    columns=["id"]
).copy()


print(
    "\nRemoving identifier variable: id"
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


expected_predictors = (
    numeric_predictors
    + categorical_predictors
)


# ------------------------------------------------------------
# Validate predictors
# ------------------------------------------------------------

missing_predictors = [
    column
    for column in expected_predictors
    if column not in X.columns
]

if missing_predictors:

    raise ValueError(
        "The following expected predictors are missing: "
        + str(missing_predictors)
    )


missing_test_predictors = [
    column
    for column in expected_predictors
    if column not in X_test_kaggle.columns
]

if missing_test_predictors:

    raise ValueError(
        "The following expected predictors are missing "
        "from test.csv: "
        + str(missing_test_predictors)
    )


X = X[
    expected_predictors
].copy()

X_test_kaggle = X_test_kaggle[
    expected_predictors
].copy()


# ------------------------------------------------------------
# Console: predictor definitions
# ------------------------------------------------------------

print(
    "\nNumeric predictors:"
)

for predictor in numeric_predictors:

    print(
        f"  {predictor}"
    )


print(
    "\nCategorical predictors:"
)

for predictor in categorical_predictors:

    print(
        f"  {predictor}"
    )


# ------------------------------------------------------------
# Train/validation split
# ------------------------------------------------------------

print(
    "\nCreating stratified train/validation split..."
)

X_train, X_validation, y_train, y_validation = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
)


print(
    f"Training observations:   {len(y_train):,}"
)

print(
    f"Validation observations: {len(y_validation):,}"
)

print(
    f"Validation proportion:   {TEST_SIZE:.0%}"
)

print(
    f"Random state:            {RANDOM_STATE}"
)


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
#
# HistGradientBoostingClassifier requires numeric input.
#
# Numeric variables are passed through unchanged.
#
# Categorical variables are ordinal encoded.
#
# The encoder is fitted ONLY on the training portion.
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
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            categorical_predictors
        )
    ],
    remainder="drop"
)


# ------------------------------------------------------------
# Fit preprocessing on training data
# ------------------------------------------------------------

print(
    "\nFitting preprocessing on training data..."
)

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_validation_processed = preprocessor.transform(
    X_validation
)

print(
    "Preprocessing completed."
)


# ------------------------------------------------------------
# Validate processed data
# ------------------------------------------------------------

if not np.isfinite(
    np.asarray(X_train_processed, dtype=float)
).all():

    raise ValueError(
        "The processed training data contains NaN or "
        "infinite values."
    )


if not np.isfinite(
    np.asarray(X_validation_processed, dtype=float)
).all():

    raise ValueError(
        "The processed validation data contains NaN or "
        "infinite values."
    )


# ------------------------------------------------------------
# Feature names
# ------------------------------------------------------------
#
# These are the ORIGINAL predictor names.
#
# There are exactly 16:
#
#   8 numeric
#   8 categorical
#
# This is intentionally different from the encoded feature
# matrix because categorical variables should receive one
# importance value as original predictors.
# ------------------------------------------------------------

feature_names = (
    numeric_predictors
    + categorical_predictors
)


# ------------------------------------------------------------
# Boosting model
# ------------------------------------------------------------

boosting_model = HistGradientBoostingClassifier(
    learning_rate=LEARNING_RATE,
    max_iter=N_ESTIMATORS,
    max_leaf_nodes=MAX_LEAF_NODES,
    l2_regularization=L2_REGULARIZATION,
    early_stopping=EARLY_STOPPING,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# Fit boosting model
# ------------------------------------------------------------

print(
    "\nFitting boosted decision-tree model..."
)

boosting_model.fit(
    X_train_processed,
    y_train
)

print(
    "Model fitting completed."
)


# ------------------------------------------------------------
# Number of boosting iterations used
# ------------------------------------------------------------

iterations_used = getattr(
    boosting_model,
    "n_iter_",
    N_ESTIMATORS
)


# ------------------------------------------------------------
# Validation predictions
# ------------------------------------------------------------

print(
    "\nGenerating validation predictions..."
)

y_pred = boosting_model.predict(
    X_validation_processed
)


# ------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_validation,
    y_pred
)

macro_f1 = f1_score(
    y_validation,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_validation,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

report_dict = classification_report(
    y_validation,
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


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

confusion = confusion_matrix(
    y_validation,
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


# ------------------------------------------------------------
# ORIGINAL-PREDICTOR PERMUTATION FEATURE IMPORTANCE
# ------------------------------------------------------------
#
# IMPORTANT:
#
# Do NOT call sklearn.inspection.permutation_importance on
# X_validation_processed here.
#
# Doing that permutes individual encoded columns rather than
# the original predictors.
#
# Instead:
#
#   1. Start with the ORIGINAL validation dataframe.
#   2. Permute one original predictor.
#   3. Re-run the fitted preprocessor.
#   4. Predict with the already-fitted boosting model.
#   5. Measure the decrease in validation accuracy.
#
# Importance is:
#
#     baseline accuracy - permuted accuracy
#
# Positive values mean that shuffling the predictor reduced
# validation accuracy.
#
# Negative values are valid and mean the shuffled predictor
# happened to produce slightly better validation accuracy.
#
# No model is refitted during this process.
# ------------------------------------------------------------

print(
    "\nCalculating permutation feature importance..."
)

print(
    "Importance is being calculated at the original "
    "predictor level."
)


# ------------------------------------------------------------
# Baseline validation accuracy
# ------------------------------------------------------------

baseline_accuracy = accuracy_score(
    y_validation,
    boosting_model.predict(
        X_validation_processed
    )
)


# ------------------------------------------------------------
# Random number generator
# ------------------------------------------------------------

rng = np.random.RandomState(
    RANDOM_STATE
)


# ------------------------------------------------------------
# Importance results
# ------------------------------------------------------------

importance_records = []


for feature in feature_names:

    print(
        f"  Permuting: {feature}"
    )

    permutation_scores = []

    for permutation_number in range(
        N_PERMUTATIONS
    ):

        X_permuted = X_validation.copy()

        shuffled_values = (
            X_permuted[feature]
            .to_numpy(copy=True)
        )

        rng.shuffle(
            shuffled_values
        )

        X_permuted.loc[
            :,
            feature
        ] = shuffled_values

        X_permuted_processed = (
            preprocessor.transform(
                X_permuted
            )
        )

        if not np.isfinite(
            np.asarray(
                X_permuted_processed,
                dtype=float
            )
        ).all():

            raise ValueError(
                "Permutation preprocessing produced "
                f"invalid values for feature '{feature}'."
            )

        permuted_predictions = (
            boosting_model.predict(
                X_permuted_processed
            )
        )

        permuted_accuracy = (
            accuracy_score(
                y_validation,
                permuted_predictions
            )
        )

        importance_value = (
            baseline_accuracy
            - permuted_accuracy
        )

        permutation_scores.append(
            importance_value
        )


    importance_records.append(
        {
            "Feature": feature,
            "Importance_Mean": float(
                np.mean(
                    permutation_scores
                )
            ),
            "Importance_SD": float(
                np.std(
                    permutation_scores,
                    ddof=0
                )
            )
        }
    )


# ------------------------------------------------------------
# Feature importance dataframe
# ------------------------------------------------------------

feature_importance_df = pd.DataFrame(
    importance_records
)


# ------------------------------------------------------------
# Validate feature importance values
# ------------------------------------------------------------

importance_columns = [
    "Importance_Mean",
    "Importance_SD"
]


for column in importance_columns:

    feature_importance_df[column] = pd.to_numeric(
        feature_importance_df[column],
        errors="coerce"
    )


if feature_importance_df[
    importance_columns
].isna().any().any():

    raise ValueError(
        "Feature importance calculation produced "
        "NaN values. The boosting script will not "
        "save invalid importance values."
    )


if not np.isfinite(
    feature_importance_df[
        importance_columns
    ].to_numpy(
        dtype=float
    )
).all():

    raise ValueError(
        "Feature importance calculation produced "
        "infinite values."
    )


# ------------------------------------------------------------
# Sort feature importance
# ------------------------------------------------------------

feature_importance_df = (
    feature_importance_df
    .sort_values(
        "Importance_Mean",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


# ------------------------------------------------------------
# Validate feature count
# ------------------------------------------------------------

if len(feature_importance_df) != len(
    feature_names
):

    raise ValueError(
        "The feature-importance table does not contain "
        "exactly one row for each original predictor."
    )


# ------------------------------------------------------------
# Top 10 feature importance
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "TOP 10 BOOSTING FEATURES"
)

print(
    "=" * 60
)

print(
    feature_importance_df
    .head(10)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Model metrics table
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    {
        "Model": [
            "Boosting"
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
        "Training_Observations": [
            len(y_train)
        ],
        "Validation_Observations": [
            len(y_validation)
        ],
        "Number_of_Classes": [
            len(class_labels)
        ],
        "Random_State": [
            RANDOM_STATE
        ],
        "Validation_Size": [
            TEST_SIZE
        ],
        "Max_Iterations": [
            N_ESTIMATORS
        ],
        "Iterations_Used": [
            iterations_used
        ],
        "Learning_Rate": [
            LEARNING_RATE
        ],
        "Max_Leaf_Nodes": [
            MAX_LEAF_NODES
        ],
        "L2_Regularization": [
            L2_REGULARIZATION
        ],
        "Early_Stopping": [
            EARLY_STOPPING
        ]
    }
)


# ------------------------------------------------------------
# Model configuration table
# ------------------------------------------------------------

configuration_df = pd.DataFrame(
    {
        "Parameter": [
            "Model",
            "Random_State",
            "Validation_Size",
            "Max_Iterations",
            "Iterations_Used",
            "Learning_Rate",
            "Max_Leaf_Nodes",
            "L2_Regularization",
            "Early_Stopping",
            "Permutation_Repeats"
        ],
        "Value": [
            "HistGradientBoostingClassifier",
            RANDOM_STATE,
            TEST_SIZE,
            N_ESTIMATORS,
            iterations_used,
            LEARNING_RATE,
            MAX_LEAF_NODES,
            L2_REGULARIZATION,
            EARLY_STOPPING,
            N_PERMUTATIONS
        ]
    }
)


# ------------------------------------------------------------
# Save validation artifacts
# ------------------------------------------------------------

metrics_df.to_csv(
    RESULT_DIR / "boosting_metrics.csv",
    index=False
)


classification_df.to_csv(
    RESULT_DIR / "boosting_classification_report.csv",
    index=False
)


confusion_df.to_csv(
    RESULT_DIR / "boosting_confusion_matrix.csv"
)


feature_importance_df.to_csv(
    RESULT_DIR / "boosting_feature_importance.csv",
    index=False
)


configuration_df.to_csv(
    RESULT_DIR / "boosting_configuration.csv",
    index=False
)


# ------------------------------------------------------------
# Console: Validation results
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "BOOSTING VALIDATION RESULTS"
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

print(
    f"Boosting iterations used: {iterations_used}"
)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print(
    "\nClassification Report:"
)

print(
    classification_df.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

print(
    "\nConfusion Matrix:"
)

print(
    confusion_df.to_string()
)


# ------------------------------------------------------------
# Final model preprocessing
# ------------------------------------------------------------
#
# The final model is fitted using all labeled training
# observations.
#
# The preprocessing object is refitted on all training data.
# ------------------------------------------------------------

print(
    "\nFitting final boosting preprocessing using all "
    "training data..."
)

final_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_predictors
        ),
        (
            "categorical",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            categorical_predictors
        )
    ],
    remainder="drop"
)


X_full_processed = final_preprocessor.fit_transform(
    X
)

X_kaggle_processed = final_preprocessor.transform(
    X_test_kaggle
)


# ------------------------------------------------------------
# Validate final processed data
# ------------------------------------------------------------

if not np.isfinite(
    np.asarray(
        X_full_processed,
        dtype=float
    )
).all():

    raise ValueError(
        "The final processed training data contains "
        "NaN or infinite values."
    )


if not np.isfinite(
    np.asarray(
        X_kaggle_processed,
        dtype=float
    )
).all():

    raise ValueError(
        "The processed Kaggle test data contains "
        "NaN or infinite values."
    )


print(
    "Final preprocessing completed."
)


# ------------------------------------------------------------
# Final boosting model
# ------------------------------------------------------------
#
# Use the number of iterations determined during validation.
#
# Early stopping is disabled because the final model is fitted
# on all available labeled training observations.
# ------------------------------------------------------------

print(
    "\nFitting final boosting model using all training data..."
)

final_boosting_model = HistGradientBoostingClassifier(
    learning_rate=LEARNING_RATE,
    max_iter=iterations_used,
    max_leaf_nodes=MAX_LEAF_NODES,
    l2_regularization=L2_REGULARIZATION,
    early_stopping=False,
    random_state=RANDOM_STATE
)


final_boosting_model.fit(
    X_full_processed,
    y
)


print(
    "Final model fitting completed."
)


# ------------------------------------------------------------
# Kaggle test predictions
# ------------------------------------------------------------

print(
    "\nGenerating predictions for Kaggle test data..."
)

kaggle_predictions = final_boosting_model.predict(
    X_kaggle_processed
)


kaggle_predictions = pd.Series(
    kaggle_predictions,
    dtype="string"
)


# ------------------------------------------------------------
# Validate predictions
# ------------------------------------------------------------

invalid_predictions = sorted(
    set(kaggle_predictions)
    - set(class_labels)
)


if invalid_predictions:

    raise ValueError(
        "The model generated invalid target classes: "
        + str(invalid_predictions)
    )


if kaggle_predictions.isna().any():

    raise ValueError(
        "The model generated missing Kaggle predictions."
    )


if len(kaggle_predictions) != len(test_ids):

    raise ValueError(
        "The number of predictions does not match "
        "the number of Kaggle test observations."
    )


# ------------------------------------------------------------
# Create Kaggle submission
# ------------------------------------------------------------
#
# The submission contains exactly:
#
#     id,NObeyesdad
# ------------------------------------------------------------

submission = pd.DataFrame(
    {
        "id": test_ids,
        "NObeyesdad": kaggle_predictions
    }
)


# ------------------------------------------------------------
# Validate submission structure
# ------------------------------------------------------------

expected_submission_columns = [
    "id",
    "NObeyesdad"
]


if submission.columns.tolist() != (
    expected_submission_columns
):

    raise ValueError(
        "Submission columns are incorrect. Expected exactly: "
        + str(expected_submission_columns)
    )


if len(submission) != len(test):

    raise ValueError(
        "Submission row count does not match test.csv."
    )


if submission[
    "NObeyesdad"
].isna().any():

    raise ValueError(
        "Submission contains missing target predictions."
    )


# ------------------------------------------------------------
# Compare submission IDs with sample submission
# ------------------------------------------------------------

if len(sample_submission) == len(submission):

    if not submission["id"].equals(
        sample_submission["id"]
    ):

        raise ValueError(
            "Submission IDs do not match the IDs in "
            "sample_submission.csv."
        )


# ------------------------------------------------------------
# Save Kaggle submission
# ------------------------------------------------------------

submission_path = (
    SUBMISSION_DIR
    / "boosting_submission.csv"
)


submission.to_csv(
    submission_path,
    index=False
)


# ------------------------------------------------------------
# Kaggle submission summary
# ------------------------------------------------------------

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
    submission["NObeyesdad"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# Final artifact confirmation
# ------------------------------------------------------------

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
        "boosting_*.csv"
    )
):

    print(
        artifact.name
    )


print(
    "boosting_submission.csv"
)


print(
    "\n12_boosting_kaggle_submission.py completed successfully."
)
