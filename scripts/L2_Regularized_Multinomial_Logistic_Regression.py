###########################################################################
# ======================================================================
# ASSIGNMENT 6
# L2-REGULARIZED MULTINOMIAL LOGISTIC REGRESSION
#
# L2_Regularized_Multinomial_Logistic_Regression.py
#
#======================================================================
#
# Purpose
# -------
# Fit and evaluate an L2-regularized multinomial logistic regression
# model for the Multi-class Prediction of Obesity Risk dataset.
# 
# This script is designed for Assignment 6 and specifically addresses
# the rubric requirement:
#
#    "Applies regularized regression"
#
# Methodology
# -----------
# 1. Load the labeled training data.
# 2. Separate predictors and target.
# 3. Encode categorical predictors using one-hot encoding.
# 4. Perform a stratified train/test split.
# 5. Standardize predictors where appropriate.
# 6. Select the regularization parameter C using cross-validation
#    on the TRAINING SET ONLY.
# 7. Fit the final L2-regularized multinomial logistic regression model
#    using the complete training split.
# 8. Evaluate the model on the held-out test set.
# 9. Calculate:
#        - Accuracy
#        - Macro F1
#        - Weighted F1
#        - Classification report
#        - Confusion matrix
# 10. Save predictions, probabilities, metrics, and diagnostics.
#
# Important
# ---------
# The held-out test set is NOT used to select the regularization
# parameter. This prevents test-set leakage.
# 
# Author: Assignment 6
# Random seed: 8555
# ======================================================================
###########################################################################

from pathlib import Path
import sys
import platform
import warnings

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ======================================================================
# CONFIGURATION
# ======================================================================

RANDOM_SEED = 8555

TEST_SIZE = 0.20

# Candidate inverse regularization strengths.
#
# Smaller C = stronger regularization.
# Larger C = weaker regularization.
#
# The best value will be selected using CV on the training split only.
C_VALUES = [
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
]

CV_FOLDS = 5

# Maximum iterations for the logistic solver.
MAX_ITER = 5000


# ======================================================================
# PROJECT PATHS
# ======================================================================

# This script assumes it is located in:
#
# C:\dev\DDS-8555\SoarDDDS8555-6\scripts\
#
# The project root is therefore one directory above this file.

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "logistic"

TRAIN_FILE = DATA_DIR / "train.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def print_header(title):
    """Print a consistent section header."""
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title):
    """Print a smaller section header."""
    print()
    print("-" * 70)
    print(title)
    print("-" * 70)


# ======================================================================
# START
# ======================================================================

print_header("L2-REGULARIZED MULTINOMIAL LOGISTIC REGRESSION")

print(
    """
Assignment 6
Applied Question #12 / Regularized Regression Requirement

Model:
    L2-Regularized Multinomial Logistic Regression

Purpose:
    Establish a regularized regression baseline for comparison
    with decision trees, bagging, random forests, boosting, and BART.

Important:
    The test set is held out until final evaluation.
    Regularization strength is selected using cross-validation
    on the training split only.
"""
)

print_subheader("CONFIGURATION")

print(f"Random seed:             {RANDOM_SEED}")
print(f"Test size:               {TEST_SIZE}")
print(f"Cross-validation folds:  {CV_FOLDS}")
print(f"Maximum iterations:      {MAX_ITER}")
print(f"Candidate C values:      {C_VALUES}")
print(f"Training file:            {TRAIN_FILE}")
print(f"Output directory:         {OUTPUT_DIR}")


# ======================================================================
# ENVIRONMENT INFORMATION
# ======================================================================

print_header("PYTHON ENVIRONMENT")

print(f"Python:       {sys.version}")
print(f"Platform:     {platform.platform()}")
print(f"NumPy:        {np.__version__}")
print(f"Pandas:       {pd.__version__}")


# ======================================================================
# LOAD DATA
# ======================================================================

print_header("LOADING DATA")

if not TRAIN_FILE.exists():
    raise FileNotFoundError(
        f"Training file was not found:\n{TRAIN_FILE}\n\n"
        "Please verify that train.csv exists in the project's data folder."
    )

print(f"Loading training data...")
df = pd.read_csv(TRAIN_FILE)

print("Training data loaded.")
print(f"Rows:    {df.shape[0]:,}")
print(f"Columns: {df.shape[1]:,}")


# ======================================================================
# VALIDATE TARGET
# ======================================================================

print_header("DATA VALIDATION")

TARGET = "NObeyesdad"

if TARGET not in df.columns:
    raise ValueError(
        f"Target column '{TARGET}' was not found in the training data."
    )

print(f"Target column: {TARGET}")

print()
print("Target classes:")

target_counts = df[TARGET].value_counts()

for class_name, count in target_counts.items():
    print(f"  {class_name:<25} {count:,}")


# ======================================================================
# IDENTIFY ID COLUMN
# ======================================================================

print_subheader("IDENTIFYING IDENTIFIER COLUMN")

if "id" in df.columns:
    ID_COLUMN = "id"
    print("Identifier column found: id")
else:
    ID_COLUMN = None
    print("No identifier column found.")


# ======================================================================
# PREPARE X AND y
# ======================================================================

print_header("PREPARING PREDICTORS AND TARGET")

y = df[TARGET].copy()

drop_columns = [TARGET]

if ID_COLUMN is not None:
    drop_columns.append(ID_COLUMN)

X = df.drop(columns=drop_columns).copy()

print(f"Predictor rows:    {X.shape[0]:,}")
print(f"Predictor columns: {X.shape[1]:,}")

print()
print("Predictor columns:")

for column in X.columns:
    print(f"  {column}")


# ======================================================================
# IDENTIFY DATA TYPES
# ======================================================================

print_header("IDENTIFYING PREDICTOR TYPES")

categorical_columns = X.select_dtypes(
    include=["str", "category", "bool"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("Categorical columns:")

for column in categorical_columns:
    print(f"  {column}")

print()
print("Numeric columns:")

for column in numeric_columns:
    print(f"  {column}")


# ======================================================================
# TRAIN / TEST SPLIT
# ======================================================================

print_header("STRATIFIED TRAIN / TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_SEED,
)

print(f"Total observations: {len(X):,}")
print(f"Training observations: {len(X_train):,}")
print(f"Test observations:     {len(X_test):,}")

print()
print("Training target distribution:")

train_counts = y_train.value_counts()

for class_name in sorted(y_train.unique()):
    print(
        f"  {class_name:<25} "
        f"{train_counts[class_name]:,}"
    )

print()
print("Test target distribution:")

test_counts = y_test.value_counts()

for class_name in sorted(y_test.unique()):
    print(
        f"  {class_name:<25} "
        f"{test_counts[class_name]:,}"
    )


# ======================================================================
# PREPROCESSING
# ======================================================================

print_header("BUILDING PREPROCESSING PIPELINE")

###########################################################################
# For numerical predictors:
#
#    Missing values → median
#    Standardization → mean 0 / standard deviation 1
#
# For categorical predictors:
#
#    Missing values → most frequent
#     One-hot encoding → binary indicator columns
#
# Scaling is important for regularized logistic regression because
# the L2 penalty acts on model coefficients. Standardizing numerical
# predictors places them on comparable scales before regularization.
###########################################################################

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=True,
            ),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns,
        ),
    ],
    remainder="drop",
)

print("Numeric preprocessing:")
print("  Median imputation")
print("  StandardScaler")

print()
print("Categorical preprocessing:")
print("  Most-frequent imputation")
print("  One-hot encoding")
print("  Unknown categories ignored")


# ======================================================================
# MODEL PIPELINE
# ======================================================================

print_header("BUILDING L2-REGULARIZED LOGISTIC MODEL")

###########################################################################
# C controls the strength of regularization.
#
# In scikit-learn:
#
#    smaller C → stronger regularization
#    larger C  → weaker regularization
#
# We use:
#
#    penalty='l2'
#
# and a multinomial-capable solver.
# 
# The GridSearchCV procedure below evaluates candidate C values
# using ONLY the training data.
###########################################################################
  
logistic_model = LogisticRegression(
    solver="lbfgs",
    max_iter=MAX_ITER,
    random_state=RANDOM_SEED,
)

model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            logistic_model,
        ),
    ]
)

print("Model:")
print("  LogisticRegression")

print()
print("Regularization:")
print("  L2")

print()
print("Solver:")
print("  lbfgs")

print()
print("Maximum iterations:")
print(f"  {MAX_ITER}")


# ======================================================================
# CROSS-VALIDATION
# ======================================================================

print_header("SELECTING REGULARIZATION STRENGTH")

print(
    """
The regularization parameter C will be selected using stratified
cross-validation on the TRAINING SET ONLY.

The held-out test set is not used during model selection.
"""
)

cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_SEED,
)

param_grid = {
    "classifier__C": C_VALUES
}

grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="accuracy",
    cv=cv,
    n_jobs=1,
    refit=True,
    return_train_score=True,
    verbose=1,
)

print("Starting cross-validation...")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    grid_search.fit(X_train, y_train)

print()
print("Cross-validation completed.")

best_C = grid_search.best_params_["classifier__C"]
best_cv_score = grid_search.best_score_

print()
print(f"Best C:                    {best_C}")
print(f"Best CV accuracy:          {best_cv_score:.6f}")


# ======================================================================
# SAVE CROSS-VALIDATION RESULTS
# ======================================================================

print_header("SAVING CROSS-VALIDATION RESULTS")

cv_results = pd.DataFrame(grid_search.cv_results_)

cv_columns = [
    "param_classifier__C",
    "mean_test_score",
    "std_test_score",
    "rank_test_score",
    "mean_train_score",
    "std_train_score",
]

available_cv_columns = [
    column
    for column in cv_columns
    if column in cv_results.columns
]

cv_output = cv_results[available_cv_columns].copy()

cv_output = cv_output.rename(
    columns={
        "param_classifier__C": "C",
        "mean_test_score": "mean_cv_accuracy",
        "std_test_score": "std_cv_accuracy",
        "rank_test_score": "cv_rank",
        "mean_train_score": "mean_training_accuracy",
        "std_train_score": "std_training_accuracy",
    }
)

cv_output["C"] = cv_output["C"].astype(float)

cv_output = cv_output.sort_values(
    by="C"
).reset_index(drop=True)

cv_file = OUTPUT_DIR / "logistic_l2_cross_validation_results.csv"

cv_output.to_csv(
    cv_file,
    index=False,
)

print(f"Saved: {cv_file}")


# ======================================================================
# FIT FINAL TRAINING MODEL
# ======================================================================

print_header("FITTING FINAL L2-REGULARIZED MODEL")

print(
    """
The selected C value was determined using only the training split.

The final model is now fitted on ALL observations in the training
split before evaluation on the untouched test set.
"""
)

final_model = grid_search.best_estimator_

final_model.fit(
    X_train,
    y_train,
)

print("Final model fitted successfully.")

print(f"Selected C: {best_C}")


# ======================================================================
# TEST PREDICTIONS
# ======================================================================

print_header("GENERATING TEST-SET PREDICTIONS")

y_pred = final_model.predict(X_test)

y_prob = final_model.predict_proba(X_test)

print("Test predictions generated.")
print(f"Predictions: {len(y_pred):,}")
print(f"Probability matrix shape: {y_prob.shape}")


# ======================================================================
# TEST METRICS
# ======================================================================

print_header("TEST-SET PERFORMANCE")

accuracy = accuracy_score(
    y_test,
    y_pred,
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0,
)

test_log_loss = log_loss(
    y_test,
    y_prob,
    labels=final_model.classes_,
)

print(f"Accuracy:       {accuracy:.6f}")
print(f"Macro F1:       {macro_f1:.6f}")
print(f"Weighted F1:    {weighted_f1:.6f}")
print(f"Log Loss:       {test_log_loss:.6f}")


# ======================================================================
# CLASSIFICATION REPORT
# ======================================================================

print_header("CLASSIFICATION REPORT")

report_dict = classification_report(
    y_test,
    y_pred,
    labels=final_model.classes_,
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report_dict).transpose()

print(
    classification_report(
        y_test,
        y_pred,
        labels=final_model.classes_,
        zero_division=0,
    )
)


# ======================================================================
# CONFUSION MATRIX
# ======================================================================

print_header("CONFUSION MATRIX")

classes = list(final_model.classes_)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes,
)

cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
)

print(cm_df)


# ======================================================================
# PREDICTION COUNTS
# ======================================================================

print_header("TEST-SET PREDICTION COUNTS")

prediction_counts = (
    pd.Series(y_pred)
    .value_counts()
    .reindex(classes, fill_value=0)
)

for class_name, count in prediction_counts.items():
    print(f"{class_name:<25} {count:,}")


# ======================================================================
# PROBABILITY VALIDATION
# ======================================================================

print_header("PROBABILITY VALIDATION")

probability_row_sums = y_prob.sum(axis=1)

print(
    f"Minimum probability row sum: "
    f"{probability_row_sums.min():.12f}"
)

print(
    f"Maximum probability row sum: "
    f"{probability_row_sums.max():.12f}"
)

print(
    f"Mean probability row sum:    "
    f"{probability_row_sums.mean():.12f}"
)

if np.allclose(
    probability_row_sums,
    1.0,
    atol=1e-10,
):
    print()
    print("SUCCESS: Test probability rows sum to 1.")
else:
    print()
    print(
        "WARNING: Some probability rows do not sum "
        "to approximately 1."
    )


# ======================================================================
# SAVE TEST PREDICTIONS
# ======================================================================

print_header("SAVING TEST PREDICTIONS")

prediction_output = pd.DataFrame(
    {
        "actual": y_test.to_numpy(),
        "predicted": y_pred,
    },
    index=X_test.index,
)

prediction_output.index.name = "original_index"

prediction_file = (
    OUTPUT_DIR
    / "logistic_l2_test_predictions.csv"
)

prediction_output.to_csv(
    prediction_file,
)

print(f"Saved: {prediction_file}")


# ======================================================================
# SAVE TEST PROBABILITIES
# ======================================================================

print_header("SAVING TEST PROBABILITIES")

probability_output = pd.DataFrame(
    y_prob,
    columns=[
        f"prob_{class_name}"
        for class_name in classes
    ],
    index=X_test.index,
)

probability_output.index.name = "original_index"

probability_file = (
    OUTPUT_DIR
    / "logistic_l2_test_probabilities.csv"
)

probability_output.to_csv(
    probability_file,
)

print(f"Saved: {probability_file}")


# ======================================================================
# SAVE METRICS
# ======================================================================

print_header("SAVING MODEL METRICS")

metrics_output = pd.DataFrame(
    {
        "model": [
            "L2-Regularized Multinomial Logistic Regression"
        ],
        "regularization": [
            "L2"
        ],
        "best_C": [
            best_C
        ],
        "cv_folds": [
            CV_FOLDS
        ],
        "best_cv_accuracy": [
            best_cv_score
        ],
        "test_accuracy": [
            accuracy
        ],
        "test_macro_f1": [
            macro_f1
        ],
        "test_weighted_f1": [
            weighted_f1
        ],
        "test_log_loss": [
            test_log_loss
        ],
        "train_rows": [
            len(X_train)
        ],
        "test_rows": [
            len(X_test)
        ],
        "random_seed": [
            RANDOM_SEED
        ],
    }
)

metrics_file = (
    OUTPUT_DIR
    / "logistic_l2_metrics.csv"
)

metrics_output.to_csv(
    metrics_file,
    index=False,
)

print(f"Saved: {metrics_file}")


# ======================================================================
# SAVE CLASSIFICATION REPORT
# ======================================================================

print_header("SAVING CLASSIFICATION REPORT")

report_file = (
    OUTPUT_DIR
    / "logistic_l2_classification_report.csv"
)

report_df.to_csv(
    report_file,
)

print(f"Saved: {report_file}")


# ======================================================================
# SAVE CONFUSION MATRIX
# ======================================================================

print_header("SAVING CONFUSION MATRIX")

confusion_file = (
    OUTPUT_DIR
    / "logistic_l2_confusion_matrix.csv"
)

cm_df.to_csv(
    confusion_file,
)

print(f"Saved: {confusion_file}")


# ======================================================================
# SAVE MODEL SUMMARY
# ======================================================================

print_header("SAVING MODEL SUMMARY")

model_summary = pd.DataFrame(
    {
        "parameter": [
            "model",
            "penalty",
            "solver",
            "best_C",
            "cv_folds",
            "best_cv_accuracy",
            "test_size",
            "random_seed",
            "max_iter",
            "train_rows",
            "test_rows",
            "number_of_classes",
            "number_of_original_predictors",
        ],
        "value": [
            "L2-Regularized Multinomial Logistic Regression",
            "L2",
            "lbfgs",
            best_C,
            CV_FOLDS,
            best_cv_score,
            TEST_SIZE,
            RANDOM_SEED,
            MAX_ITER,
            len(X_train),
            len(X_test),
            len(classes),
            X.shape[1],
        ],
    }
)

summary_file = (
    OUTPUT_DIR
    / "logistic_l2_model_summary.csv"
)

model_summary.to_csv(
    summary_file,
    index=False,
)

print(f"Saved: {summary_file}")


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print_header("FINAL L2-REGULARIZED LOGISTIC REGRESSION SUMMARY")

print()
print("MODEL")
print("L2-Regularized Multinomial Logistic Regression")

print()
print("REGULARIZATION")
print(f"Selected C:       {best_C}")
print(f"CV folds:         {CV_FOLDS}")
print(f"CV accuracy:      {best_cv_score:.6f}")

print()
print("HELD-OUT TEST PERFORMANCE")
print(f"Accuracy:         {accuracy:.6f}")
print(f"Macro F1:         {macro_f1:.6f}")
print(f"Weighted F1:      {weighted_f1:.6f}")
print(f"Log Loss:         {test_log_loss:.6f}")

print()
print("DATA SPLIT")
print(f"Training rows:    {len(X_train):,}")
print(f"Test rows:        {len(X_test):,}")

print()
print("OUTPUT FILES")
print(f"Cross-validation:     {cv_file}")
print(f"Predictions:          {prediction_file}")
print(f"Probabilities:        {probability_file}")
print(f"Metrics:              {metrics_file}")
print(f"Classification report: {report_file}")
print(f"Confusion matrix:     {confusion_file}")
print(f"Model summary:        {summary_file}")

print()
print("=" * 70)
print("L2-REGULARIZED MULTINOMIAL LOGISTIC REGRESSION COMPLETE")
print("=" * 70)
