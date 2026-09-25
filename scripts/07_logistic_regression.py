# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 07_logistic_regression.py
#
# Purpose:
#   Fit and evaluate a multinomial logistic regression model
#   for the NObeyesdad target.
#
#   This script:
#     1. Loads the training data.
#     2. Separates predictors and target.
#     3. Removes the identifier variable.
#     4. Identifies numeric and categorical predictors.
#     5. Builds a preprocessing pipeline.
#     6. Standardizes numeric predictors.
#     7. One-hot encodes categorical predictors.
#     8. Fits multinomial logistic regression.
#     9. Evaluates the model on a held-out test set.
#    10. Saves durable results for Report.Rmd.
#
#   Logistic regression is used as the non-tree baseline
#   for comparison with Decision Tree, Bagging, Random Forest,
#   Boosting, and BART.
# ============================================================


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)


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

MAX_ITER = 2000


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print(
    "\nLoading training data..."
)

train = pd.read_csv(
    DATA_DIR / "train.csv"
)

print(
    f"Training observations: {len(train):,}"
)

print(
    f"Training variables:    {len(train.columns):,}"
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
# Separate predictors and target
# ------------------------------------------------------------

X = train.drop(
    columns=[TARGET]
)

y = train[TARGET].astype(str)


# ------------------------------------------------------------
# Remove identifier from predictors
# ------------------------------------------------------------

if "id" in X.columns:

    print(
        "\nRemoving identifier variable: id"
    )

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


X = X[
    expected_predictors
]


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
# Train/test split
# ------------------------------------------------------------
#
# The same random state, test size, and stratification
# are used by the other Assignment 6 model scripts.
# ------------------------------------------------------------

print(
    "\nCreating stratified train/test split..."
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)


print(
    f"Training observations: {len(y_train):,}"
)

print(
    f"Test observations:     {len(y_test):,}"
)

print(
    f"Test proportion:       {TEST_SIZE:.0%}"
)

print(
    f"Random state:          {RANDOM_STATE}"
)


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
#
# Numeric predictors are standardized using parameters
# estimated from the training data only.
#
# Categorical predictors are one-hot encoded.
#
# Standardization improves numerical conditioning for
# logistic-regression optimization and does not introduce
# information from the held-out test observations.
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
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
# Logistic regression model
# ------------------------------------------------------------
#
# NObeyesdad contains seven classes.
#
# The current scikit-learn implementation uses the
# appropriate multiclass formulation for this problem.
#
# max_iter is retained at 2,000 to provide sufficient
# optimization iterations.
# ------------------------------------------------------------

logistic_model = LogisticRegression(
    max_iter=MAX_ITER,
    random_state=RANDOM_STATE
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
            logistic_model
        )
    ]
)


# ------------------------------------------------------------
# Fit model
# ------------------------------------------------------------

print(
    "\nFitting multinomial logistic regression model..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Model fitting completed."
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

print(
    "\nGenerating held-out test predictions..."
)

y_pred = model.predict(
    X_test
)


# ------------------------------------------------------------
# Performance metrics
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

class_labels = sorted(
    y.unique()
)

report_dict = classification_report(
    y_test,
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
    y_test,
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
# Model metrics table
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    {
        "Model": [
            "Logistic Regression"
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
        "Test_Observations": [
            len(y_test)
        ],
        "Training_Observations": [
            len(y_train)
        ],
        "Number_of_Classes": [
            len(class_labels)
        ],
        "Random_State": [
            RANDOM_STATE
        ],
        "Test_Size": [
            TEST_SIZE
        ],
        "Max_Iterations": [
            MAX_ITER
        ]
    }
)


# ------------------------------------------------------------
# Logistic regression coefficients
# ------------------------------------------------------------
#
# Coefficients are extracted after the pipeline has been
# fitted.
#
# Because the numeric variables were standardized, their
# coefficients correspond to a one-standard-deviation
# increase in the respective numeric predictor, conditional
# on the fitted model.
#
# Categorical coefficients correspond to the one-hot encoded
# indicator variables relative to the encoding reference
# categories.
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


coefficient_df = pd.DataFrame(
    classifier.coef_,
    columns=feature_names
)


coefficient_df.insert(
    0,
    "Class",
    classifier.classes_
)


# ------------------------------------------------------------
# Save durable artifacts
# ------------------------------------------------------------

metrics_df.to_csv(
    RESULT_DIR / "logistic_regression_metrics.csv",
    index=False
)


classification_df.to_csv(
    RESULT_DIR / "logistic_regression_classification_report.csv",
    index=False
)


confusion_df.to_csv(
    RESULT_DIR / "logistic_regression_confusion_matrix.csv"
)


coefficient_df.to_csv(
    RESULT_DIR / "logistic_regression_coefficients.csv",
    index=False
)


# ------------------------------------------------------------
# Console output
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "MULTINOMIAL LOGISTIC REGRESSION RESULTS"
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
    f"Training observations: {len(y_train):,}"
)


print(
    f"Test observations:     {len(y_test):,}"
)


print(
    f"Number of classes:     {len(class_labels)}"
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
# Artifact confirmation
# ------------------------------------------------------------

print(
    "\nRESULT ARTIFACTS SAVED"
)

print(
    "=" * 60
)


for artifact in sorted(
    RESULT_DIR.glob(
        "logistic_regression_*.csv"
    )
):

    print(
        artifact.name
    )


print(
    "\n07_logistic_regression.py completed successfully."
)
