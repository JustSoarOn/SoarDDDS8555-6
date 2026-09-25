# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 09_decision_tree_kaggle_submission.py
#
# Purpose:
#   Fit and evaluate a Decision Tree classifier for the
#   NObeyesdad target in the Kaggle
#   Multi-class Prediction of Obesity Risk dataset.
#
#   This script:
#     1. Loads the training, test, and sample-submission data.
#     2. Separates predictors and target.
#     3. Removes the identifier variable.
#     4. Defines numeric and categorical predictors.
#     5. Creates a reproducible stratified 80/20 split.
#     6. Builds a preprocessing pipeline.
#     7. Fits a Decision Tree classifier.
#     8. Evaluates performance on the held-out validation set.
#     9. Saves metrics, classification report, confusion matrix,
#        configuration, and feature importance.
#    10. Creates a visualization of the fitted decision tree.
#    11. Fits the final tree using all available training data.
#    12. Generates predictions for the complete Kaggle test set.
#    13. Creates a Kaggle-ready prediction file.
#
#   The validation set is used only for model evaluation.
#   The final model is subsequently fitted to all labeled
#   training observations before predicting the Kaggle test set.
# ============================================================


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

RESULT_DIR = OUTPUT_DIR / "results"
FIG_DIR = OUTPUT_DIR / "figures"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
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

MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 5

CRITERION = "gini"


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

print("=" * 70)
print("DECISION TREE")
print("=" * 70)

print("\nLoading data...")

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
# Validate required files and columns
# ------------------------------------------------------------

if TARGET not in train.columns:

    raise ValueError(
        f"Target variable '{TARGET}' was not found "
        "in train.csv."
    )


if TARGET in test.columns:

    raise ValueError(
        f"The test data should not contain the target "
        f"variable '{TARGET}'."
    )


required_submission_columns = [
    "id",
    TARGET
]

if list(sample_submission.columns) != required_submission_columns:

    raise ValueError(
        "sample_submission.csv must contain exactly "
        "the columns: ['id', 'NObeyesdad']"
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
# Validate predictor columns
# ------------------------------------------------------------

missing_train_predictors = [
    column
    for column in expected_predictors
    if column not in train.columns
]

missing_test_predictors = [
    column
    for column in expected_predictors
    if column not in test.columns
]


if missing_train_predictors:

    raise ValueError(
        "The following predictors are missing from train.csv: "
        + str(missing_train_predictors)
    )


if missing_test_predictors:

    raise ValueError(
        "The following predictors are missing from test.csv: "
        + str(missing_test_predictors)
    )


# ------------------------------------------------------------
# Separate predictors and target
# ------------------------------------------------------------

X = train[
    expected_predictors
]

y = train[
    TARGET
].astype(str)


X_test_kaggle = test[
    expected_predictors
]


# ------------------------------------------------------------
# Target classes
# ------------------------------------------------------------

class_labels = sorted(
    y.unique()
)

print("\nTarget classes:")

for label in class_labels:

    print(
        f"  {label}"
    )


# ------------------------------------------------------------
# Console: predictor definitions
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Train / validation split
# ------------------------------------------------------------
#
# A stratified split is used so that the class distribution
# is represented in both the training and validation data.
#
# random_state=42 keeps the split reproducible and consistent
# with the other Assignment 6 models.
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
#
# Numeric variables are passed through unchanged.
#
# Categorical variables are one-hot encoded. The fitted
# encoder learns its categories from the training portion only.
#
# This preprocessing is contained inside the modeling pipeline
# so that the validation data are not used to estimate
# preprocessing parameters.
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
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_predictors
        )
    ]
)


# ------------------------------------------------------------
# Decision Tree model
# ------------------------------------------------------------
#
# max_depth and min_samples_leaf provide modest regularization.
# These restrictions make the tree easier to interpret and
# reduce the likelihood of fitting very small terminal nodes.
# ------------------------------------------------------------

decision_tree_model = DecisionTreeClassifier(
    criterion=CRITERION,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE
)


# ------------------------------------------------------------
# Complete validation modeling pipeline
# ------------------------------------------------------------

validation_model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            decision_tree_model
        )
    ]
)


# ------------------------------------------------------------
# Fit validation model
# ------------------------------------------------------------

print(
    "\nFitting decision tree model..."
)

validation_model.fit(
    X_train,
    y_train
)

print(
    "Model fitting completed."
)


# ------------------------------------------------------------
# Validation predictions
# ------------------------------------------------------------

print(
    "\nGenerating validation predictions..."
)

y_pred = validation_model.predict(
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
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_valid,
    y_pred,
    average="weighted",
    zero_division=0
)


# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Validation metrics artifact
# ------------------------------------------------------------

metrics_df = pd.DataFrame(
    {
        "Model": [
            "Decision Tree"
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
        "Number_of_Classes": [
            len(class_labels)
        ],
        "Random_State": [
            RANDOM_STATE
        ],
        "Validation_Proportion": [
            TEST_SIZE
        ]
    }
)


# ------------------------------------------------------------
# Feature importance
# ------------------------------------------------------------
#
# Feature importance is calculated from the fitted validation
# tree. Because categorical predictors are one-hot encoded,
# individual category indicators appear as separate features.
# ------------------------------------------------------------

validation_classifier = validation_model.named_steps[
    "classifier"
]

validation_preprocessor = validation_model.named_steps[
    "preprocessor"
]

feature_names = (
    validation_preprocessor
    .get_feature_names_out()
)


importance_df = pd.DataFrame(
    {
        "Feature": feature_names,
        "Importance": (
            validation_classifier
            .feature_importances_
        )
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
# Model configuration artifact
# ------------------------------------------------------------

configuration_df = pd.DataFrame(
    {
        "Parameter": [
            "Model",
            "Criterion",
            "Maximum Depth",
            "Minimum Samples per Leaf",
            "Random State",
            "Validation Proportion",
            "Training Observations",
            "Validation Observations",
            "Numeric Predictors",
            "Categorical Predictors",
            "Predictor Encoding",
            "Identifier Excluded"
        ],
        "Value": [
            "DecisionTreeClassifier",
            CRITERION,
            MAX_DEPTH,
            MIN_SAMPLES_LEAF,
            RANDOM_STATE,
            TEST_SIZE,
            len(y_train),
            len(y_valid),
            ", ".join(numeric_predictors),
            ", ".join(categorical_predictors),
            "One-hot encoding for categorical predictors",
            "Yes"
        ]
    }
)


# ------------------------------------------------------------
# Save validation artifacts
# ------------------------------------------------------------

metrics_df.to_csv(
    RESULT_DIR / "decision_tree_metrics.csv",
    index=False
)


classification_df.to_csv(
    RESULT_DIR / "decision_tree_classification_report.csv",
    index=False
)


confusion_df.to_csv(
    RESULT_DIR / "decision_tree_confusion_matrix.csv"
)


importance_df.to_csv(
    RESULT_DIR / "decision_tree_feature_importance.csv",
    index=False
)


configuration_df.to_csv(
    RESULT_DIR / "decision_tree_configuration.csv",
    index=False
)


# ------------------------------------------------------------
# Console: validation results
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "DECISION TREE VALIDATION RESULTS"
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
    "\nClassification Report:"
)

print(
    classification_df.to_string(
        index=False
    )
)


print(
    "\nConfusion Matrix:"
)

print(
    confusion_df.to_string()
)


# ------------------------------------------------------------
# Top validation feature importance
# ------------------------------------------------------------

print(
    "\nTOP 10 TREE FEATURES"
)

print(
    "=" * 60
)

print(
    importance_df
    .head(10)
    .to_string(
        index=False
    )
)


# ------------------------------------------------------------
# Decision-tree visualization
# ------------------------------------------------------------
#
# The visualization uses the validation-fitted tree. This is
# appropriate for interpreting the model that produced the
# reported validation results.
# ------------------------------------------------------------

print(
    "\nCreating decision-tree visualization..."
)


plt.figure(
    figsize=(26, 16)
)


plot_tree(
    validation_classifier,
    feature_names=feature_names,
    class_names=validation_classifier.classes_,
    filled=True,
    rounded=True,
    proportion=True,
    impurity=True,
    fontsize=7
)


plt.title(
    "Decision Tree for NObeyesdad Classification",
    fontsize=16
)


plt.tight_layout()


plt.savefig(
    FIG_DIR / "decision_tree.png",
    dpi=200,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------
# Final model for Kaggle prediction
# ------------------------------------------------------------
#
# After validation performance has been evaluated, a separate
# final model is fitted using ALL labeled observations.
#
# This model is NOT used to calculate the validation metrics
# above. It is used only to generate predictions for the
# unlabeled Kaggle test set.
# ------------------------------------------------------------

print(
    "\nFitting final decision tree using all training data..."
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
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_predictors
        )
    ]
)


final_tree = DecisionTreeClassifier(
    criterion=CRITERION,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE
)


final_model = Pipeline(
    steps=[
        (
            "preprocessor",
            final_preprocessor
        ),
        (
            "classifier",
            final_tree
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


# ------------------------------------------------------------
# Generate Kaggle test predictions
# ------------------------------------------------------------

print(
    "\nGenerating predictions for Kaggle test data..."
)


test_predictions = final_model.predict(
    X_test_kaggle
)


# ------------------------------------------------------------
# Validate predictions
# ------------------------------------------------------------

if len(test_predictions) != len(test):

    raise ValueError(
        "The number of test predictions does not match "
        "the number of rows in test.csv."
    )


prediction_values = set(
    test_predictions
)

valid_class_values = set(
    class_labels
)


unexpected_predictions = (
    prediction_values
    - valid_class_values
)


if unexpected_predictions:

    raise ValueError(
        "Unexpected class labels were generated: "
        + str(unexpected_predictions)
    )


# ------------------------------------------------------------
# Create Kaggle-ready submission
# ------------------------------------------------------------
#
# The sample submission provides the required ID ordering and
# column structure. Only the NObeyesdad values are replaced
# with predictions from the final model.
# ------------------------------------------------------------

submission = sample_submission.copy()

submission[TARGET] = test_predictions

submission = submission[
    [
        "id",
        TARGET
    ]
]


# ------------------------------------------------------------
# Submission validation
# ------------------------------------------------------------

if len(submission) != len(sample_submission):

    raise ValueError(
        "Submission row count does not match "
        "sample_submission.csv."
    )


if len(submission) != len(test):

    raise ValueError(
        "Submission row count does not match test.csv."
    )


if not submission["id"].equals(
    sample_submission["id"]
):

    raise ValueError(
        "Submission IDs do not match the IDs in "
        "sample_submission.csv."
    )


if not submission["id"].equals(
    test["id"]
):

    raise ValueError(
        "Submission IDs do not match the IDs in test.csv."
    )


if submission[TARGET].isna().any():

    raise ValueError(
        "Submission contains missing NObeyesdad predictions."
    )


if list(submission.columns) != [
    "id",
    "NObeyesdad"
]:

    raise ValueError(
        "Submission must contain exactly "
        "['id', 'NObeyesdad']."
    )


# ------------------------------------------------------------
# Save Kaggle submission
# ------------------------------------------------------------

submission_path = (
    SUBMISSION_DIR
    / "decision_tree_submission.csv"
)


submission.to_csv(
    submission_path,
    index=False
)


# ------------------------------------------------------------
# Submission confirmation
# ------------------------------------------------------------

print(
    "\nKAGGLE SUBMISSION"
)

print(
    "=" * 60
)

print(
    f"Rows:    {len(submission):,}"
)

print(
    f"Columns: {list(submission.columns)}"
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


print(
    "decision_tree_metrics.csv"
)

print(
    "decision_tree_classification_report.csv"
)

print(
    "decision_tree_confusion_matrix.csv"
)

print(
    "decision_tree_feature_importance.csv"
)

print(
    "decision_tree_configuration.csv"
)

print(
    "decision_tree.png"
)

print(
    "decision_tree_submission.csv"
)


print(
    "\n09_decision_tree_kaggle_submission.py completed successfully."
)

