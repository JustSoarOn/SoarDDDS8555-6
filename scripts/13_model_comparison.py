# ============================================================
# Assignment 6: Build and Evaluate Tree Models
#
# 13_model_comparison.py
#
# Purpose:
#   Consolidate the validation results from the four required
#   tree-based classification models:
#
#     1. Decision Tree
#     2. Bagging
#     3. Random Forest
#     4. Boosting
#
#   This script:
#     1. Reads the durable result artifacts created by
#        09_decision_tree.py, 10_bagging.py,
#        11_random_forest.py, and 12_boosting.py.
#     2. Creates one model-performance comparison table.
#     3. Creates a class-level F1 comparison table.
#     4. Creates a feature-importance comparison table.
#     5. Creates a model configuration table.
#     6. Creates report-ready performance figures.
#
#   IMPORTANT:
#     No predictive models are fitted in this script.
#     No Kaggle predictions are generated in this script.
#
#   All model results are based on the same stratified
#   80/20 validation split using random_state=42.
#
#   FEATURE-IMPORTANCE HANDLING:
#     Decision Tree, Bagging, and Random Forest use the
#     "Importance" column from their feature-importance
#     artifacts.
#
#     Boosting uses "Importance_Mean" from its permutation-
#     importance artifact.
#
#     Importance values are converted to numeric, +/- infinity
#     is converted to NaN, and invalid/missing values are
#     replaced with 0.0 before sorting, saving, or plotting.
# ============================================================


# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "output"

RESULT_DIR = OUTPUT_DIR / "results"

FIG_DIR = OUTPUT_DIR / "figures"

RESULT_DIR.mkdir(
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

MODEL_FILES = {
    "Decision Tree": {
        "metrics": "decision_tree_metrics.csv",
        "classification": "decision_tree_classification_report.csv",
        "importance": "decision_tree_feature_importance.csv",
        "configuration": "decision_tree_configuration.csv"
    },
    "Bagging": {
        "metrics": "bagging_metrics.csv",
        "classification": "bagging_classification_report.csv",
        "importance": "bagging_feature_importance.csv",
        "configuration": "bagging_configuration.csv"
    },
    "Random Forest": {
        "metrics": "random_forest_metrics.csv",
        "classification": "random_forest_classification_report.csv",
        "importance": "random_forest_feature_importance.csv",
        "configuration": "random_forest_configuration.csv"
    },
    "Boosting": {
        "metrics": "boosting_metrics.csv",
        "classification": "boosting_classification_report.csv",
        "importance": "boosting_feature_importance.csv",
        "configuration": "boosting_configuration.csv"
    }
}


MODEL_ORDER = [
    "Decision Tree",
    "Bagging",
    "Random Forest",
    "Boosting"
]


# ------------------------------------------------------------
# Console header
# ------------------------------------------------------------

print("=" * 70)

print(
    "MODEL COMPARISON"
)

print("=" * 70)


# ------------------------------------------------------------
# Validate required artifacts
# ------------------------------------------------------------

print(
    "\nChecking required model artifacts..."
)


missing_files = []


for model_name, files in MODEL_FILES.items():

    for file_type, filename in files.items():

        file_path = RESULT_DIR / filename

        if not file_path.exists():

            missing_files.append(
                str(file_path)
            )


if missing_files:

    print(
        "\nERROR: The following required artifacts "
        "were not found:"
    )

    for filename in missing_files:

        print(
            f"  {filename}"
        )

    raise FileNotFoundError(
        "Required model result artifacts are missing. "
        "Run scripts 09 through 12 before running "
        "13_model_comparison.py."
    )


print(
    "All required model artifacts found."
)


# ------------------------------------------------------------
# Read model metrics
# ------------------------------------------------------------

print(
    "\nReading model performance metrics..."
)


metric_rows = []


for model_name in MODEL_ORDER:

    filename = MODEL_FILES[
        model_name
    ]["metrics"]

    metrics_path = RESULT_DIR / filename

    metrics = pd.read_csv(
        metrics_path
    )

    if metrics.empty:

        raise ValueError(
            f"{filename} contains no observations."
        )

    row = metrics.iloc[0].to_dict()

    row["Model"] = model_name

    metric_rows.append(
        row
    )


metrics_df = pd.DataFrame(
    metric_rows
)


# ------------------------------------------------------------
# Select and order performance columns
# ------------------------------------------------------------

preferred_metric_columns = [
    "Model",
    "Accuracy",
    "Macro_F1",
    "Weighted_F1",
    "Training_Observations",
    "Validation_Observations",
    "Test_Observations",
    "Number_of_Classes",
    "Random_State",
    "Validation_Size",
    "Test_Size"
]


available_metric_columns = [
    column
    for column in preferred_metric_columns
    if column in metrics_df.columns
]


model_performance = metrics_df[
    available_metric_columns
].copy()


model_performance["Model"] = pd.Categorical(
    model_performance["Model"],
    categories=MODEL_ORDER,
    ordered=True
)


model_performance = (
    model_performance
    .sort_values("Model")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Save model performance comparison
# ------------------------------------------------------------

model_performance_path = (
    RESULT_DIR
    / "model_performance_comparison.csv"
)


model_performance.to_csv(
    model_performance_path,
    index=False
)


# ------------------------------------------------------------
# Read classification reports
# ------------------------------------------------------------

print(
    "\nReading class-level classification results..."
)


classification_rows = []


for model_name in MODEL_ORDER:

    filename = MODEL_FILES[
        model_name
    ]["classification"]

    classification_path = (
        RESULT_DIR / filename
    )

    classification = pd.read_csv(
        classification_path
    )

    if classification.empty:

        raise ValueError(
            f"{filename} contains no observations."
        )

    if "Class" not in classification.columns:

        raise ValueError(
            f"{filename} does not contain the required "
            "'Class' column."
        )

    classification = classification[
        ~classification["Class"].isin(
            [
                "accuracy",
                "macro avg",
                "weighted avg"
            ]
        )
    ].copy()

    classification.insert(
        0,
        "Model",
        model_name
    )

    classification_rows.append(
        classification
    )


class_performance = pd.concat(
    classification_rows,
    ignore_index=True
)


class_performance["Model"] = pd.Categorical(
    class_performance["Model"],
    categories=MODEL_ORDER,
    ordered=True
)


class_performance = (
    class_performance
    .sort_values(
        [
            "Model",
            "Class"
        ]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# Save class-level comparison
# ------------------------------------------------------------

class_performance_path = (
    RESULT_DIR
    / "class_performance_comparison.csv"
)


class_performance.to_csv(
    class_performance_path,
    index=False
)


# ------------------------------------------------------------
# Create F1 comparison table
# ------------------------------------------------------------

f1_comparison = (
    class_performance[
        [
            "Model",
            "Class",
            "f1-score"
        ]
    ]
    .pivot(
        index="Class",
        columns="Model",
        values="f1-score"
    )
    .reset_index()
)


for model_name in MODEL_ORDER:

    if model_name not in f1_comparison.columns:

        f1_comparison[model_name] = pd.NA


f1_comparison = f1_comparison[
    [
        "Class"
    ]
    + MODEL_ORDER
]


f1_comparison_path = (
    RESULT_DIR
    / "f1_score_comparison_by_class.csv"
)


f1_comparison.to_csv(
    f1_comparison_path,
    index=False
)


# ------------------------------------------------------------
# Read feature importance results
# ------------------------------------------------------------

print(
    "\nReading feature-importance results..."
)


importance_rows = []


for model_name in MODEL_ORDER:

    filename = MODEL_FILES[
        model_name
    ]["importance"]

    importance_path = (
        RESULT_DIR / filename
    )

    importance = pd.read_csv(
        importance_path
    )

    if importance.empty:

        raise ValueError(
            f"{filename} contains no observations."
        )

    importance.insert(
        0,
        "Model",
        model_name
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Decision Tree, Bagging, and Random Forest use their
    # standard tree-based feature importance stored in
    # "Importance".
    #
    # Boosting uses permutation importance stored in
    # "Importance_Mean".
    # --------------------------------------------------------

    if model_name == "Boosting":

        if "Importance_Mean" not in importance.columns:

            raise ValueError(
                f"{filename} does not contain the required "
                "'Importance_Mean' column for Boosting "
                "permutation importance."
            )

        importance["Importance_Mean"] = (
            importance["Importance_Mean"]
        )

        print(
            "Using 'Importance_Mean' permutation importance "
            "for Boosting."
        )

    else:

        if "Importance" not in importance.columns:

            raise ValueError(
                f"{filename} does not contain the required "
                "'Importance' column for {model_name}."
            )

        importance["Importance_Mean"] = (
            importance["Importance"]
        )

        print(
            f"Using 'Importance' feature importance "
            f"for {model_name}."
        )

    importance_rows.append(
        importance
    )


feature_importance = pd.concat(
    importance_rows,
    ignore_index=True
)


feature_importance["Model"] = pd.Categorical(
    feature_importance["Model"],
    categories=MODEL_ORDER,
    ordered=True
)


# ------------------------------------------------------------
# Validate feature column
# ------------------------------------------------------------

if "Feature" not in feature_importance.columns:

    raise ValueError(
        "The model importance artifacts must contain "
        "a 'Feature' column."
    )


feature_importance["Feature"] = (
    feature_importance["Feature"]
    .astype(str)
)


# ------------------------------------------------------------
# Clean feature-importance values
# ------------------------------------------------------------
#
# Convert importance values to numeric.
#
# Convert +/- infinity to NaN.
#
# Replace invalid/missing values with 0.0.
#
# This protection is applied after selecting the correct
# importance column for each model.
# ------------------------------------------------------------

feature_importance["Importance_Mean"] = pd.to_numeric(
    feature_importance["Importance_Mean"],
    errors="coerce"
)


feature_importance["Importance_Mean"] = (
    feature_importance["Importance_Mean"]
    .replace(
        [
            float("inf"),
            float("-inf")
        ],
        pd.NA
    )
)


invalid_importance_count = (
    feature_importance["Importance_Mean"]
    .isna()
    .sum()
)


if invalid_importance_count > 0:

    print(
        "\nWARNING:"
    )

    print(
        f"Found {invalid_importance_count} invalid or "
        "missing feature-importance value(s)."
    )

    print(
        "Replacing invalid importance values with 0.0 "
        "so the comparison table and figure remain valid."
    )

    feature_importance["Importance_Mean"] = (
        feature_importance["Importance_Mean"]
        .fillna(0.0)
    )


# ------------------------------------------------------------
# Validate cleaned feature importance
# ------------------------------------------------------------

if not pd.api.types.is_numeric_dtype(
    feature_importance["Importance_Mean"]
):

    raise TypeError(
        "Importance_Mean must be numeric after cleaning."
    )


if not feature_importance[
    "Importance_Mean"
].notna().all():

    raise ValueError(
        "Feature importance still contains NaN values "
        "after cleaning."
    )


if not feature_importance[
    "Importance_Mean"
].apply(
    lambda value: pd.notna(value)
    and value != float("inf")
    and value != float("-inf")
).all():

    raise ValueError(
        "Feature importance contains non-finite values "
        "after cleaning."
    )


# ------------------------------------------------------------
# Save complete feature-importance comparison
# ------------------------------------------------------------

feature_importance = (
    feature_importance
    .sort_values(
        [
            "Model",
            "Importance_Mean"
        ],
        ascending=[
            True,
            False
        ]
    )
    .reset_index(drop=True)
)


feature_importance_path = (
    RESULT_DIR
    / "feature_importance_comparison.csv"
)


feature_importance.to_csv(
    feature_importance_path,
    index=False
)


# ------------------------------------------------------------
# Create top-feature comparison
# ------------------------------------------------------------

top_feature_rows = []


for model_name in MODEL_ORDER:

    model_importance = (
        feature_importance[
            feature_importance["Model"]
            == model_name
        ]
        .copy()
        .sort_values(
            "Importance_Mean",
            ascending=False
        )
        .head(10)
    )

    model_importance["Rank"] = range(
        1,
        len(model_importance) + 1
    )

    top_feature_rows.append(
        model_importance[
            [
                "Model",
                "Rank",
                "Feature",
                "Importance_Mean"
            ]
        ]
    )


if top_feature_rows:

    top_features = pd.concat(
        top_feature_rows,
        ignore_index=True
    )

else:

    top_features = pd.DataFrame(
        columns=[
            "Model",
            "Rank",
            "Feature",
            "Importance_Mean"
        ]
    )


# ------------------------------------------------------------
# Final top-feature validation
# ------------------------------------------------------------

if not top_features.empty:

    top_features["Importance_Mean"] = (
        pd.to_numeric(
            top_features["Importance_Mean"],
            errors="coerce"
        )
        .replace(
            [
                float("inf"),
                float("-inf")
            ],
            pd.NA
        )
        .fillna(0.0)
    )


top_features_path = (
    RESULT_DIR
    / "top_10_features_by_model.csv"
)


top_features.to_csv(
    top_features_path,
    index=False
)


# ------------------------------------------------------------
# Read model configurations
# ------------------------------------------------------------

print(
    "\nReading model configurations..."
)


configuration_rows = []


for model_name in MODEL_ORDER:

    filename = MODEL_FILES[
        model_name
    ]["configuration"]

    configuration_path = (
        RESULT_DIR / filename
    )

    configuration = pd.read_csv(
        configuration_path
    )

    if configuration.empty:

        raise ValueError(
            f"{filename} contains no observations."
        )

    configuration.insert(
        0,
        "Model",
        model_name
    )

    configuration_rows.append(
        configuration
    )


configuration_df = pd.concat(
    configuration_rows,
    ignore_index=True
)


configuration_path = (
    RESULT_DIR
    / "model_configuration_comparison.csv"
)


configuration_df.to_csv(
    configuration_path,
    index=False
)


# ------------------------------------------------------------
# Performance summary
# ------------------------------------------------------------

performance_long = model_performance[
    [
        "Model",
        "Accuracy",
        "Macro_F1",
        "Weighted_F1"
    ]
].copy()


performance_long_path = (
    RESULT_DIR
    / "model_performance_summary.csv"
)


performance_long.to_csv(
    performance_long_path,
    index=False
)


# ------------------------------------------------------------
# Create performance figure
# ------------------------------------------------------------

print(
    "\nCreating model-performance figure..."
)


plot_data = model_performance.copy()

plot_data["Model"] = (
    plot_data["Model"]
    .astype(str)
)


# Convert plotting metrics to numeric and protect
# the figure from invalid values.

for column in [
    "Accuracy",
    "Macro_F1",
    "Weighted_F1"
]:

    if column in plot_data.columns:

        plot_data[column] = (
            pd.to_numeric(
                plot_data[column],
                errors="coerce"
            )
            .replace(
                [
                    float("inf"),
                    float("-inf")
                ],
                pd.NA
            )
            .fillna(0.0)
        )


x = range(
    len(plot_data)
)

width = 0.25


fig, ax = plt.subplots(
    figsize=(11, 6)
)


ax.bar(
    [i - width for i in x],
    plot_data["Accuracy"],
    width=width,
    label="Accuracy",
    color="#4472C4"
)


ax.bar(
    list(x),
    plot_data["Macro_F1"],
    width=width,
    label="Macro F1",
    color="#ED7D31"
)


ax.bar(
    [i + width for i in x],
    plot_data["Weighted_F1"],
    width=width,
    label="Weighted F1",
    color="#70AD47"
)


ax.set_xticks(
    list(x)
)

ax.set_xticklabels(
    plot_data["Model"]
)


ax.set_ylim(
    0.75,
    0.95
)


ax.set_ylabel(
    "Validation Performance"
)


ax.set_xlabel(
    "Model"
)


ax.set_title(
    "Validation Performance of Tree-Based Classification Models"
)


ax.legend()


ax.grid(
    axis="y",
    alpha=0.25
)


for spine in [
    "top",
    "right"
]:

    ax.spines[
        spine
    ].set_visible(False)


plt.tight_layout()


performance_figure_path = (
    FIG_DIR
    / "model_performance_comparison.png"
)


plt.savefig(
    performance_figure_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------
# Create top-feature figure
# ------------------------------------------------------------

print(
    "\nCreating feature-importance figure..."
)


top_plot = top_features.copy()


fig, axes = plt.subplots(
    2,
    2,
    figsize=(14, 10)
)


axes = axes.flatten()


for index, model_name in enumerate(
    MODEL_ORDER
):

    ax = axes[index]

    model_data = (
        top_plot[
            top_plot["Model"]
            == model_name
        ]
        .copy()
        .sort_values(
            "Importance_Mean",
            ascending=True
        )
    )

    # --------------------------------------------------------
    # Additional protection against NaN/infinite plotting
    # values.
    # --------------------------------------------------------

    model_data["Importance_Mean"] = (
        pd.to_numeric(
            model_data["Importance_Mean"],
            errors="coerce"
        )
        .replace(
            [
                float("inf"),
                float("-inf")
            ],
            pd.NA
        )
        .fillna(0.0)
    )


    if model_data.empty:

        ax.text(
            0.5,
            0.5,
            "No feature-importance data",
            ha="center",
            va="center",
            transform=ax.transAxes
        )

        ax.set_title(
            model_name
        )

        ax.set_xlabel(
            "Feature Importance"
        )

        continue


    ax.barh(
        model_data["Feature"],
        model_data["Importance_Mean"],
        color="#4472C4"
    )

    ax.set_title(
        model_name
    )

    ax.set_xlabel(
        "Feature Importance"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )


    for spine in [
        "top",
        "right"
    ]:

        ax.spines[
            spine
        ].set_visible(False)


fig.suptitle(
    "Top Predictive Features by Tree-Based Model",
    fontsize=15
)


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.96
    ]
)


feature_figure_path = (
    FIG_DIR
    / "model_feature_importance_comparison.png"
)


plt.savefig(
    feature_figure_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


# ------------------------------------------------------------
# Console: Model performance
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "MODEL PERFORMANCE COMPARISON"
)

print(
    "=" * 60
)


display_columns = [
    "Model",
    "Accuracy",
    "Macro_F1",
    "Weighted_F1"
]


print(
    model_performance[
        display_columns
    ]
    .round(4)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Console: Class F1 comparison
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "F1 SCORE BY CLASS"
)

print(
    "=" * 60
)


print(
    f1_comparison
    .round(4)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Console: Top features
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 60
)

print(
    "TOP 10 FEATURES BY MODEL"
)

print(
    "=" * 60
)


for model_name in MODEL_ORDER:

    print(
        f"\n{model_name}:"
    )

    model_top = (
        top_features[
            top_features["Model"]
            == model_name
        ]
        .copy()
    )

    if model_top.empty:

        print(
            "  No feature-importance observations available."
        )

        continue


    print(
        model_top[
            [
                "Rank",
                "Feature",
                "Importance_Mean"
            ]
        ]
        .round(6)
        .to_string(index=False)
    )


# ------------------------------------------------------------
# Artifact confirmation
# ------------------------------------------------------------

print(
    "\n"
    + "=" * 70
)

print(
    "MODEL COMPARISON ARTIFACTS SAVED"
)

print(
    "=" * 70
)


print(
    "\nResult files:"
)


comparison_files = [
    RESULT_DIR / "model_performance_comparison.csv",
    RESULT_DIR / "class_performance_comparison.csv",
    RESULT_DIR / "f1_score_comparison_by_class.csv",
    RESULT_DIR / "feature_importance_comparison.csv",
    RESULT_DIR / "model_configuration_comparison.csv",
    RESULT_DIR / "model_performance_summary.csv",
    RESULT_DIR / "top_10_features_by_model.csv"
]


for artifact in comparison_files:

    if artifact.exists():

        print(
            f"  {artifact.name}"
        )


print(
    "\nFigure files:"
)


for artifact in [
    FIG_DIR / "model_performance_comparison.png",
    FIG_DIR / "model_feature_importance_comparison.png"
]:

    if artifact.exists():

        print(
            f"  {artifact.name}"
        )


print(
    "\n13_model_comparison.py completed successfully."
)
