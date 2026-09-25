# ============================================================
# Assignment 6
# Bayesian Additive Regression Trees (BART)
#
# 08_bart.py - main BART model/evaluation script 
#   using an 80/20 train/test split, with results saved under output/results/
#
#     RANDOM_STATE = 42 for the split and RANDOM_SEED = 8555
#
# PURPOSE
#   Fit and evaluate a computationally manageable BART
#   classification model for the NObeyesdad obesity
#   classification problem.
#
# APPROACH
#   One-vs-rest BART:
#
#       One binary BART model is fitted for each target class.
#
#       For each class:
#           1 = current class
#           0 = all other classes
#
#   Multiclass prediction:
#       Obtain a posterior probability for every class,
#       normalize the probabilities across classes, and
#       predict the class with the largest probability.
#
# OUTPUT
#   Results are written to:
#
#       output/results/
#
#   including:
#
#       bart_metrics.csv
#       bart_classification_report.csv
#       bart_confusion_matrix.csv
#       bart_model_summary.csv
#       bart_test_probabilities.csv
#       bart_feature_metadata.csv
#       bart_comparison.csv
#       bart_completed.txt
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import gc
import multiprocessing as mp
import os
import sys
import warnings

import numpy as np
import pandas as pd

import pymc as pm
import pymc_bart as pmb

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # REPRODUCIBILITY
    # ========================================================

    RANDOM_SEED = 8555
    RANDOM_STATE = 42

    np.random.seed(RANDOM_SEED)


    # ========================================================
    # PROJECT PATHS
    # ========================================================

    PROJECT_DIR = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
        )
    )

    DATA_DIR = os.path.join(
        PROJECT_DIR,
        "data",
    )

    RESULT_DIR = os.path.join(
        PROJECT_DIR,
        "output",
        "results",
    )

    os.makedirs(
        RESULT_DIR,
        exist_ok=True,
    )

    COMPLETION_FILE = os.path.join(
        RESULT_DIR,
        "bart_completed.txt",
    )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    TARGET = "NObeyesdad"

    TEST_SIZE = 0.20

    # --------------------------------------------------------
    # Computationally manageable BART settings.
    #
    # If the assignment requires a faster test run, these can
    # be reduced. If more computation is available, they can
    # later be increased.
    # --------------------------------------------------------

    BART_TREES = 50
    BART_DRAWS = 500
    BART_TUNE = 500

    BART_CHAINS = 4
    BART_CORES = 2


    # ========================================================
    # WARNING HANDLING
    # ========================================================

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )


    # ========================================================
    # REMOVE OLD COMPLETION MARKER
    # ========================================================

    if os.path.exists(COMPLETION_FILE):

        os.remove(
            COMPLETION_FILE
        )


    # ========================================================
    # HEADER
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "BAYESIAN ADDITIVE REGRESSION TREES (BART)"
    )

    print("=" * 70)

    print(
        "ASSIGNMENT 6"
    )

    print("\nConfiguration:")

    print(
        f"Random seed:       {RANDOM_SEED}"
    )

    print(
        f"Random state:      {RANDOM_STATE}"
    )

    print(
        f"BART trees:        {BART_TREES}"
    )

    print(
        f"Posterior draws:   {BART_DRAWS}"
    )

    print(
        f"Tuning draws:      {BART_TUNE}"
    )

    print(
        f"Chains:            {BART_CHAINS}"
    )

    print(
        f"Cores:             {BART_CORES}"
    )


    # ========================================================
    # ENVIRONMENT INFORMATION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "PYTHON ENVIRONMENT"
    )

    print("=" * 70)

    print(
        f"Python:     {sys.version}"
    )

    print(
        f"Executable: {sys.executable}"
    )

    print(
        f"NumPy:      {np.__version__}"
    )

    print(
        f"Pandas:     {pd.__version__}"
    )

    print(
        f"PyMC:       {pm.__version__}"
    )

    print(
        f"PyMC-BART:  {pmb.__version__}"
    )


    # ========================================================
    # LOAD DATA
    # ========================================================

    train_file = os.path.join(
        DATA_DIR,
        "train.csv",
    )

    if not os.path.exists(train_file):

        raise FileNotFoundError(
            "\nTraining data not found:\n"
            f"{train_file}"
        )

    print(
        "\nLoading training data...",
        flush=True,
    )

    train = pd.read_csv(
        train_file
    )

    print(
        "Training data loaded.",
        flush=True,
    )

    print(
        f"Training rows:    {len(train):,}",
        flush=True,
    )

    print(
        f"Training columns: {len(train.columns):,}",
        flush=True,
    )


    # ========================================================
    # TARGET
    # ========================================================

    if TARGET not in train.columns:

        raise ValueError(
            f"Target '{TARGET}' was not found in train.csv."
        )

    X = train.drop(
        columns=[TARGET]
    )

    y = train[TARGET].astype(str)


    # ========================================================
    # REMOVE IDENTIFIER
    # ========================================================

    if "id" in X.columns:

        print(
            "\nRemoving identifier variable: id",
            flush=True,
        )

        X = X.drop(
            columns=["id"]
        )


    # ========================================================
    # PREDICTOR DEFINITIONS
    # ========================================================

    numeric_predictors = [
        "Age",
        "Height",
        "Weight",
        "FCVC",
        "NCP",
        "CH2O",
        "FAF",
        "TUE",
    ]

    categorical_predictors = [
        "Gender",
        "family_history_with_overweight",
        "FAVC",
        "CAEC",
        "SMOKE",
        "SCC",
        "CALC",
        "MTRANS",
    ]

    expected_predictors = (
        numeric_predictors
        + categorical_predictors
    )


    # ========================================================
    # CHECK PREDICTORS
    # ========================================================

    missing_predictors = [
        column
        for column in expected_predictors
        if column not in X.columns
    ]

    if missing_predictors:

        raise ValueError(
            "Missing expected predictors:\n"
            + str(missing_predictors)
        )

    X = X[
        expected_predictors
    ]


    # ========================================================
    # CLASS LABELS
    # ========================================================

    class_labels = sorted(
        y.unique()
    )

    n_classes = len(
        class_labels
    )

    print(
        "\nTarget classes:"
    )

    for label in class_labels:

        print(
            f"  {label}"
        )

    print(
        f"\nNumber of classes: {n_classes}"
    )


    # ========================================================
    # TRAIN / TEST SPLIT
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "TRAIN / TEST SPLIT"
    )

    print("=" * 70)

    (
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"Training observations: {len(y_train):,}"
    )

    print(
        f"Test observations:     {len(y_test):,}"
    )

    print(
        f"Test proportion:        {TEST_SIZE:.0%}"
    )


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    print(
        "\nEncoding categorical predictors...",
        flush=True,
    )

    # --------------------------------------------------------
    # Compatible with current scikit-learn versions.
    # --------------------------------------------------------

    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:

        # ----------------------------------------------------
        # Compatibility with older scikit-learn versions.
        # ----------------------------------------------------

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )

    X_train_categorical = encoder.fit_transform(
        X_train_raw[categorical_predictors]
    )

    X_test_categorical = encoder.transform(
        X_test_raw[categorical_predictors]
    )


    # ========================================================
    # NUMERIC VARIABLES
    # ========================================================

    X_train_numeric = (
        X_train_raw[
            numeric_predictors
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    X_test_numeric = (
        X_test_raw[
            numeric_predictors
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    # ========================================================
    # FEATURE NAMES
    # ========================================================

    encoded_categorical_names = (
        encoder.get_feature_names_out(
            categorical_predictors
        )
    )

    feature_names = (
        list(numeric_predictors)
        + list(encoded_categorical_names)
    )


    # ========================================================
    # COMBINE FEATURES
    # ========================================================

    X_train = np.column_stack(
        [
            X_train_numeric,
            X_train_categorical,
        ]
    ).astype(
        np.float64
    )

    X_test = np.column_stack(
        [
            X_test_numeric,
            X_test_categorical,
        ]
    ).astype(
        np.float64
    )


    # ========================================================
    # DATA VALIDATION
    # ========================================================

    if not np.isfinite(
        X_train
    ).all():

        raise ValueError(
            "X_train contains non-finite values."
        )

    if not np.isfinite(
        X_test
    ).all():

        raise ValueError(
            "X_test contains non-finite values."
        )

    print(
        "\nEncoded training matrix:"
    )

    print(
        f"Rows:    {X_train.shape[0]:,}"
    )

    print(
        f"Columns: {X_train.shape[1]:,}"
    )

    print(
        "\nEncoded test matrix:"
    )

    print(
        f"Rows:    {X_test.shape[0]:,}"
    )

    print(
        f"Columns: {X_test.shape[1]:,}"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    bart_probability_matrix = np.zeros(
        (
            len(y_test),
            n_classes,
        ),
        dtype=np.float64,
    )

    model_summary_rows = []


    # ========================================================
    # FIT ONE-VS-REST BART MODELS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "FITTING ONE-VS-REST BART MODELS"
    )

    print("=" * 70)

    print(
        "\nOnly held-out test predictions will be generated."
    )

    print(
        "Training posterior predictions are not generated."
    )


    # ========================================================
    # CLASS LOOP
    # ========================================================

    for class_number, class_label in enumerate(
        class_labels,
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"MODEL {class_number} OF {n_classes}"
        )

        print(
            f"Target class: {class_label}"
        )

        print("-" * 70)


        # ====================================================
        # ONE-VS-REST TARGET
        # ====================================================

        y_train_binary = (
            y_train == class_label
        ).astype(
            np.int8
        ).to_numpy()

        positive_count = int(
            y_train_binary.sum()
        )

        negative_count = int(
            len(y_train_binary)
            - positive_count
        )

        print(
            f"Positive observations: {positive_count:,}"
        )

        print(
            f"Negative observations: {negative_count:,}"
        )


        # ====================================================
        # RANDOM SEEDS
        # ====================================================

        model_seed = (
            RANDOM_SEED
            + class_number
        )

        prediction_seed = (
            RANDOM_SEED
            + 100
            + class_number
        )


        # ====================================================
        # BUILD AND FIT BART MODEL
        # ====================================================

        print(
            "\nBuilding BART model...",
            flush=True,
        )

        with pm.Model() as bart_model:

            # ------------------------------------------------
            # Predictor container.
            # ------------------------------------------------

            X_data = pm.Data(
                "X_data",
                X_train,
            )

            # ------------------------------------------------
            # BART probability model.
            # ------------------------------------------------

            p = pmb.BART(
                "p",
                X=X_data,
                Y=y_train_binary,
                m=BART_TREES,
            )

            # ------------------------------------------------
            # Binary likelihood.
            # ------------------------------------------------

            pm.Bernoulli(
                "y",
                p=p,
                observed=y_train_binary,
            )


            # =================================================
            # POSTERIOR SAMPLING
            # =================================================

            print(
                "\nStarting posterior sampling...",
                flush=True,
            )

            trace = pm.sample(
                draws=BART_DRAWS,
                tune=BART_TUNE,
                chains=BART_CHAINS,
                cores=BART_CORES,
                random_seed=model_seed,
                progressbar=True,
                return_inferencedata=True,
            )

            print(
                "\nPosterior sampling completed.",
                flush=True,
            )


            # =================================================
            # SWITCH TO TEST DATA
            # =================================================

            print(
                "\nSwitching model to held-out test data...",
                flush=True,
            )

            pm.set_data(
                {
                    "X_data": X_test
                }
            )


            # =================================================
            # OUT-OF-SAMPLE PREDICTION
            # =================================================

            print(
                "Generating held-out test predictions...",
                flush=True,
            )

            prediction_idata = (
                pm.sample_posterior_predictive(
                    trace,
                    var_names=["p"],
                    sample_vars=["p"],
                    predictions=True,
                    random_seed=prediction_seed,
                    progressbar=True,
                    return_inferencedata=True,
                )
            )



        # ====================================================
        # VERIFY PREDICTIONS
        # ====================================================

        if not hasattr(
            prediction_idata,
            "predictions",
        ):

            raise ValueError(
                "BART prediction output does not contain "
                "a predictions group."
            )

        if "p" not in prediction_idata.predictions:

            raise ValueError(
                "BART predictions do not contain variable 'p'."
            )

        p_samples = (
            prediction_idata
            .predictions["p"]
        )

        print(
            f"\nPrediction dimensions: {p_samples.dims}",
            flush=True,
        )

        print(
            f"Prediction shape: {p_samples.shape}",
            flush=True,
        )


        # ====================================================
        # IDENTIFY SAMPLE DIMENSIONS
        # ====================================================

        sample_dims = {
            "chain",
            "draw",
        }

        observation_dims = [
            dimension
            for dimension in p_samples.dims
            if dimension not in sample_dims
        ]

        if len(observation_dims) != 1:

            raise ValueError(
                "Unexpected BART prediction dimensions.\n"
                f"Dimensions: {p_samples.dims}\n"
                f"Shape: {p_samples.shape}"
            )

        observation_dim = observation_dims[0]

        print(
            f"Observation dimension: {observation_dim}",
            flush=True,
        )


        # ====================================================
        # POSTERIOR MEAN PROBABILITY
        # ====================================================

        test_probability = (
            p_samples
            .mean(
                dim=[
                    "chain",
                    "draw",
                ]
            )
            .values
        )

        test_probability = np.asarray(
            test_probability,
            dtype=np.float64,
        )


        # ====================================================
        # REMOVE ONLY SINGLETON DIMENSIONS
        # ====================================================

        test_probability = np.squeeze(
            test_probability
        )


        # ====================================================
        # VALIDATE SHAPE
        # ====================================================

        if test_probability.ndim != 1:

            raise ValueError(
                "BART test probability array did not reduce "
                "to one dimension.\n"
                f"Shape: {test_probability.shape}\n"
                f"Expected: ({len(y_test)},)"
            )

        if len(test_probability) != len(y_test):

            raise ValueError(
                "Unexpected number of BART test predictions.\n"
                f"Class: {class_label}\n"
                f"Expected: {len(y_test)}\n"
                f"Received: {len(test_probability)}"
            )


        # ====================================================
        # NUMERICAL SAFETY
        # ====================================================

        test_probability = np.nan_to_num(
            test_probability,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

        test_probability = np.clip(
            test_probability,
            0.0,
            1.0,
        )


        # ====================================================
        # STORE PROBABILITIES
        # ====================================================

        bart_probability_matrix[
            :,
            class_number - 1,
        ] = test_probability

        print(
            f"Stored {len(test_probability):,} "
            f"probabilities for '{class_label}'.",
            flush=True,
        )


        # ====================================================
        # STORE MODEL SUMMARY
        # ====================================================

        model_summary_rows.append(
            {
                "Class": class_label,
                "Positive_Training_Observations": positive_count,
                "Negative_Training_Observations": negative_count,
                "Trees": BART_TREES,
                "Posterior_Draws": BART_DRAWS,
                "Tuning_Draws": BART_TUNE,
                "Chains": BART_CHAINS,
                "Cores": BART_CORES,
                "Random_Seed": model_seed,
            }
        )


        print(
            f"\nCompleted BART model for: {class_label}",
            flush=True,
        )


        # ====================================================
        # CLEANUP
        # ====================================================

        del trace
        del prediction_idata
        del bart_model

        gc.collect()


    # ========================================================
    # ALL MODELS COMPLETED
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "ALL ONE-VS-REST BART MODELS COMPLETED"
    )

    print("=" * 70)


    # ========================================================
    # RAW PROBABILITY VALIDATION
    # ========================================================

    if not np.isfinite(
        bart_probability_matrix
    ).all():

        raise ValueError(
            "BART probability matrix contains non-finite values."
        )

    if (
        bart_probability_matrix < 0
    ).any() or (
        bart_probability_matrix > 1
    ).any():

        raise ValueError(
            "BART probability matrix contains values outside "
            "the [0, 1] interval."
        )


    # ========================================================
    # NORMALIZE ONE-VS-REST PROBABILITIES
    # ========================================================

    print(
        "\nNormalizing class probabilities...",
        flush=True,
    )

    probability_sum = (
        bart_probability_matrix
        .sum(
            axis=1,
            keepdims=True,
        )
    )

    # --------------------------------------------------------
    # One-vs-rest probabilities are independently estimated.
    # Therefore, they do not necessarily sum to one.
    #
    # Normalize them to obtain a multiclass probability vector.
    # --------------------------------------------------------

    bart_probability_matrix = (
        bart_probability_matrix
        / np.clip(
            probability_sum,
            1e-12,
            None,
        )
    )


    # ========================================================
    # NORMALIZED PROBABILITY CHECK
    # ========================================================

    normalized_sums = (
        bart_probability_matrix
        .sum(
            axis=1
        )
    )

    if not np.allclose(
        normalized_sums,
        1.0,
        atol=1e-6,
    ):

        raise ValueError(
            "Normalized BART probabilities do not sum to 1."
        )


    # ========================================================
    # MULTICLASS PREDICTIONS
    # ========================================================

    print(
        "Generating multiclass predictions...",
        flush=True,
    )

    prediction_indices = np.argmax(
        bart_probability_matrix,
        axis=1,
    )

    y_pred = np.array(
        [
            class_labels[index]
            for index in prediction_indices
        ]
    )


    # ========================================================
    # PERFORMANCE METRICS
    # ========================================================

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


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )

    classification_df = (
        pd.DataFrame(
            report_dict
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "Class"
            }
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    confusion = confusion_matrix(
        y_test,
        y_pred,
        labels=class_labels,
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=class_labels,
        columns=class_labels,
    )

    confusion_df.index.name = "Actual"

    confusion_df.columns.name = "Predicted"


    # ========================================================
    # METRICS TABLE
    # ========================================================

    metrics_df = pd.DataFrame(
        {
            "Model": ["BART"],
            "Accuracy": [accuracy],
            "Macro_F1": [macro_f1],
            "Weighted_F1": [weighted_f1],
            "Test_Observations": [len(y_test)],
            "Training_Observations": [len(y_train)],
            "Number_of_Classes": [n_classes],
            "Trees_Per_BART_Model": [BART_TREES],
            "Posterior_Draws": [BART_DRAWS],
            "Tuning_Draws": [BART_TUNE],
            "Chains": [BART_CHAINS],
            "Cores": [BART_CORES],
            "Test_Size": [TEST_SIZE],
            "Random_State": [RANDOM_STATE],
            "Random_Seed": [RANDOM_SEED],
        }
    )


    # ========================================================
    # TEST PROBABILITY TABLE
    # ========================================================

    probability_columns = [
        f"Probability_{label}"
        for label in class_labels
    ]

    probability_df = pd.DataFrame(
        bart_probability_matrix,
        columns=probability_columns,
    )

    probability_df.insert(
        0,
        "Actual",
        y_test.to_numpy(),
    )

    probability_df.insert(
        1,
        "Predicted",
        y_pred,
    )


    # ========================================================
    # MODEL SUMMARY
    # ========================================================

    model_summary_df = pd.DataFrame(
        model_summary_rows
    )


    # ========================================================
    # FEATURE METADATA
    # ========================================================

    feature_metadata_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Feature_Number": np.arange(
                1,
                len(feature_names) + 1,
            ),
        }
    )


    # ========================================================
    # COMPARISON TABLE
    # ========================================================

    comparison_df = pd.DataFrame(
        {
            "Model": ["BART"],
            "Accuracy": [accuracy],
            "Macro_F1": [macro_f1],
            "Weighted_F1": [weighted_f1],
        }
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "SAVING BART RESULTS"
    )

    print("=" * 70)


    metrics_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_metrics.csv",
        ),
        index=False,
    )


    classification_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_classification_report.csv",
        ),
        index=False,
    )


    confusion_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_confusion_matrix.csv",
        )
    )


    model_summary_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_model_summary.csv",
        ),
        index=False,
    )


    probability_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_test_probabilities.csv",
        ),
        index=False,
    )


    feature_metadata_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_feature_metadata.csv",
        ),
        index=False,
    )


    comparison_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_comparison.csv",
        ),
        index=False,
    )


    # ========================================================
    # CONSOLE RESULTS
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "BART RESULTS"
    )

    print("=" * 70)

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


    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    print(
        "\nClassification Report:"
    )

    print(
        classification_df.to_string(
            index=False
        )
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print(
        "\nConfusion Matrix:"
    )

    print(
        confusion_df.to_string()
    )


    # ========================================================
    # ARTIFACT CONFIRMATION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "BART RESULT ARTIFACTS"
    )

    print("=" * 70)

    bart_artifacts = sorted(
        [
            artifact
            for artifact in os.listdir(
                RESULT_DIR
            )
            if artifact.startswith(
                "bart_"
            )
        ]
    )

    for artifact in bart_artifacts:

        print(
            artifact
        )


    # ========================================================
    # COMPLETION MARKER
    # ========================================================

    with open(
        COMPLETION_FILE,
        "w",
        encoding="utf-8",
    ) as marker:

        marker.write(
            "08_bart.py completed successfully.\n"
        )

        marker.write(
            f"Python: {sys.version}\n"
        )

        marker.write(
            f"NumPy: {np.__version__}\n"
        )

        marker.write(
            f"Pandas: {pd.__version__}\n"
        )

        marker.write(
            f"PyMC: {pm.__version__}\n"
        )

        marker.write(
            f"PyMC-BART: {pmb.__version__}\n"
        )

        marker.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        marker.write(
            f"Macro_F1: {macro_f1:.6f}\n"
        )

        marker.write(
            f"Weighted_F1: {weighted_f1:.6f}\n"
        )

        marker.write(
            f"Training_Observations: {len(y_train)}\n"
        )

        marker.write(
            f"Test_Observations: {len(y_test)}\n"
        )

        marker.write(
            f"Number_of_Classes: {n_classes}\n"
        )

        marker.write(
            f"Trees: {BART_TREES}\n"
        )

        marker.write(
            f"Posterior_Draws: {BART_DRAWS}\n"
        )

        marker.write(
            f"Tuning_Draws: {BART_TUNE}\n"
        )

        marker.write(
            f"Chains: {BART_CHAINS}\n"
        )

        marker.write(
            f"Cores: {BART_CORES}\n"
        )

        marker.write(
            f"Test_Size: {TEST_SIZE}\n"
        )

        marker.write(
            f"Random_State: {RANDOM_STATE}\n"
        )

        marker.write(
            f"Random_Seed: {RANDOM_SEED}\n"
        )


    # ========================================================
    # FINAL CONFIRMATION
    # ========================================================

    print("\n" + "=" * 70)

    print(
        "08_bart.py COMPLETED SUCCESSFULLY"
    )

    print("=" * 70)

    print(
        f"\nResults saved to:\n{RESULT_DIR}"
    )


# ============================================================
# WINDOWS MULTIPROCESSING ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()
