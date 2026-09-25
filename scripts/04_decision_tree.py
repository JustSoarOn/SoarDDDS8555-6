# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
#     04_decision_tree.py
#
# Decision Tree classifier for the Kaggle
# Multi-class Prediction of Obesity Risk dataset.
#
# Outputs:
#    output/results/decision_tree_metrics.csv
#    output/results/decision_tree_classification_report.csv
#    output/results/decision_tree_confusion_matrix.csv
#    output/results/decision_tree_configuration.csv
#    output/figures/decision_tree.png
#
# The script uses the same 80/20 stratified split (random_state=42)
# used by the other Assignment 6 models.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RESULT_DIR = PROJECT_ROOT / "output" / "results"
FIG_DIR = PROJECT_ROOT / "output" / "figures"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOCATION
# ============================================================

TRAIN_PATH = DATA_DIR / "train.csv"

if not TRAIN_PATH.exists():
    raise FileNotFoundError(
        f"Training data were not found at:\n{TRAIN_PATH}\n\n"
        "Place train.csv in the data directory before running "
        "decision_tree.py."
    )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DECISION TREE")
print("=" * 70)

print("\nLoading training data...")

df = pd.read_csv(TRAIN_PATH)

print(f"Training rows: {df.shape[0]:,}")
print(f"Training columns: {df.shape[1]}")


# ============================================================
# TARGET AND PREDICTORS
# ============================================================

TARGET = "NObeyesdad"

if TARGET not in df.columns:
    raise ValueError(
        f"Target variable '{TARGET}' was not found in the training data."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]


# The Kaggle ID is an identifier rather than a substantive predictor.
# It is therefore excluded from model fitting.
if "id" in X.columns:
    X = X.drop(columns=["id"])


numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\nCategorical predictors:")
print(categorical_features)

print("\nNumeric predictors:")
print(numeric_features)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\n" + "=" * 30)
print("TRAIN / VALIDATION SPLIT")
print("=" * 30)

print(f"Training observations:   {len(X_train):,}")
print(f"Validation observations: {len(X_valid):,}")
print(f"Validation proportion:   {TEST_SIZE:.0%}")
print(f"Random state:            {RANDOM_STATE}")


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ],
    remainder="drop",
)


# ============================================================
# DECISION TREE
# ============================================================

# A modest depth restriction provides a more interpretable tree
# and helps control overfitting.
MAX_DEPTH = 6
MIN_SAMPLES_LEAF = 5

tree_model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    random_state=RANDOM_STATE,
)


model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "classifier",
            tree_model,
        ),
    ]
)


# ============================================================
# FIT MODEL
# ============================================================

print("\nFitting decision tree model...")

model.fit(
    X_train,
    y_train,
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_valid)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_valid,
    y_pred,
)

macro_f1 = f1_score(
    y_valid,
    y_pred,
    average="macro",
)

weighted_f1 = f1_score(
    y_valid,
    y_pred,
    average="weighted",
)


print("\n" + "=" * 30)
print("DECISION TREE RESULTS")
print("=" * 30)

print(f"Accuracy:     {accuracy:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_valid,
    y_pred,
    output_dict=True,
    zero_division=0,
)

report_df = (
    pd.DataFrame(report_dict)
    .transpose()
    .reset_index()
    .rename(columns={"index": "Class"})
)

print("\nClassification Report:")
print(
    classification_report(
        y_valid,
        y_pred,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

classes = sorted(y.unique())

cm = confusion_matrix(
    y_valid,
    y_pred,
    labels=classes,
)

confusion_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes,
)

confusion_df.index.name = "Actual"
confusion_df.columns.name = "Predicted"


# ============================================================
# METRICS ARTIFACT
# ============================================================

metrics_df = pd.DataFrame(
    {
        "Model": ["Decision Tree"],
        "Accuracy": [accuracy],
        "Macro_F1": [macro_f1],
        "Weighted_F1": [weighted_f1],
    }
)

metrics_df.to_csv(
    RESULT_DIR / "decision_tree_metrics.csv",
    index=False,
)


# ============================================================
# CLASSIFICATION REPORT ARTIFACT
# ============================================================

report_df.to_csv(
    RESULT_DIR / "decision_tree_classification_report.csv",
    index=False,
)


# ============================================================
# CONFUSION MATRIX ARTIFACT
# ============================================================

confusion_df.to_csv(
    RESULT_DIR / "decision_tree_confusion_matrix.csv"
)


# ============================================================
# CONFIGURATION ARTIFACT
# ============================================================

configuration_df = pd.DataFrame(
    {
        "Parameter": [
            "Model",
            "Criterion",
            "Max Depth",
            "Minimum Samples per Leaf",
            "Random State",
            "Training Observations",
            "Validation Observations",
            "Validation Proportion",
            "Predictor Encoding",
            "ID Excluded",
        ],
        "Value": [
            "DecisionTreeClassifier",
            "Gini",
            MAX_DEPTH,
            MIN_SAMPLES_LEAF,
            RANDOM_STATE,
            len(X_train),
            len(X_valid),
            TEST_SIZE,
            "One-hot encoding for categorical predictors",
            "Yes",
        ],
    }
)

configuration_df.to_csv(
    RESULT_DIR / "decision_tree_configuration.csv",
    index=False,
)


# ============================================================
# TREE VISUALIZATION
# ============================================================

print("\nCreating decision-tree visualization...")

fitted_preprocessor = model.named_steps["preprocessor"]
fitted_tree = model.named_steps["classifier"]

try:
    feature_names = fitted_preprocessor.get_feature_names_out()
except Exception:
    feature_names = [
        f"Feature_{i}"
        for i in range(
            fitted_tree.n_features_in_
        )
    ]


plt.figure(
    figsize=(24, 14)
)

plot_tree(
    fitted_tree,
    feature_names=feature_names,
    class_names=classes,
    filled=True,
    rounded=True,
    proportion=True,
    impurity=True,
    fontsize=7,
)

plt.title(
    "Decision Tree for NObeyesdad Classification",
    fontsize=16,
)

plt.tight_layout()

plt.savefig(
    FIG_DIR / "decision_tree.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("RESULT ARTIFACTS SAVED")
print("=" * 70)

print("decision_tree_metrics.csv")
print("decision_tree_classification_report.csv")
print("decision_tree_confusion_matrix.csv")
print("decision_tree_configuration.csv")
print("decision_tree.png")

print("\nDecision tree analysis completed successfully.")
