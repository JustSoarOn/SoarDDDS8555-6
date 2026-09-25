# ============================================================
# Assignment 6
# Bayesian Additive Regression Trees (BART)
#
# 14_bart_validation_andevaluation.py
#
# PURPOSE
#   Fit and evaluate BART classification models using an
#   80/20 stratified training/validation split.
#
# IMPORTANT
#   PyMC-BART is a latent regression function and its raw BART
#   output is not restricted to [0, 1].
#
#   Therefore, for binary classification:
#
#       mu = BART(...)
#       p  = sigmoid(mu)
#       y  = Bernoulli(p=p)
#
#   One-vs-rest BART is used for multiclass classification.
#
#   For each target class:
#
#       1 = current class
#       0 = all other classes
#
#   The posterior mean probability for every class is obtained
#   on the held-out validation set.
#
#   Because one-vs-rest probabilities do not necessarily sum
#   to exactly 1, they are normalized across classes before
#   generating the final multiclass prediction.
#
# OUTPUT
#   output/results/
#
#       bart_metrics.csv
#       bart_classification_report.csv
#       bart_confusion_matrix.csv
#       bart_model_summary.csv
#       bart_test_probabilities.csv
#       bart_feature_metadata.csv
#       bart_comparison.csv
#       bart_configuration.csv
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

    np.random.seed(
        RANDOM_SEED
    )


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

    BART_TREES = 10

    BART_DRAWS = 100

    BART_TUNE = 100

    BART_CHAINS = 1

    BART_CORES = 1


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

    if os.path.exists(
        COMPLETION_FILE
    ):

        os.remove(
            COMPLETION_FILE
        )


    # ========================================================
    # HEADER
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BART VALIDATION AND EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        "ASSIGNMENT 6"
    )

    print(
        "\nThis script fits BART classification models using "
        "an 80/20 training/validation split."
    )

    print(
        "It evaluates validation performance and saves "
        "durable results."
    )


    # ========================================================
    # CONFIGURATION DISPLAY
    # ========================================================

    print(
        "\nConfiguration:"
    )

    print(
        f"Random seed:       {RANDOM_SEED}"
    )

    print(
        f"Random state:      {RANDOM_STATE}"
    )

    print(
        f"Validation size:   {TEST_SIZE:.0%}"
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

    print(
        "\n" + "=" * 70
    )

    print(
        "PYTHON ENVIRONMENT"
    )

    print(
        "=" * 70
    )

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

    if not os.path.exists(
        train_file
    ):

        raise FileNotFoundError(
            "\nTraining data not found:\n"
            f"{train_file}"
        )


    print(
        "\n" + "=" * 70
    )

    print(
        "LOADING DATA"
    )

    print(
        "=" * 70
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

    y = train[
        TARGET
    ].astype(
        str
    )


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
        "\n" + "=" * 70
    )

    print(
        "TARGET CLASSES"
    )

    print(
        "=" * 70
    )


    for label in class_labels:

        print(
            f"  {label}"
        )


    print(
        f"\nNumber of classes: {n_classes}"
    )


    # ========================================================
    # TARGET DISTRIBUTION
    # ========================================================

    print(
        "\nTraining target distribution:"
    )

    print(
        y.value_counts()
        .sort_index()
    )


    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAIN / VALIDATION SPLIT"
    )

    print(
        "=" * 70
    )


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
        f"\nTotal observations:       {len(y):,}"
    )

    print(
        f"Training observations:    {len(y_train):,}"
    )

    print(
        f"Validation observations:  {len(y_test):,}"
    )

    print(
        f"Validation proportion:    {TEST_SIZE:.2%}"
    )


    # ========================================================
    # ONE-HOT ENCODING
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ENCODING PREDICTORS"
    )

    print(
        "=" * 70
    )

    print(
        "\nEncoding categorical predictors...",
        flush=True,
    )


    try:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

    except TypeError:

        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse=False,
        )


    X_train_categorical = encoder.fit_transform(
        X_train_raw[
            categorical_predictors
        ]
    )


    X_test_categorical = encoder.transform(
        X_test_raw[
            categorical_predictors
        ]
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
        "\nEncoded validation matrix:"
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

    print(
        "\n" + "=" * 70
    )

    print(
        "FITTING ONE-VS-REST BART MODELS"
    )

    print(
        "=" * 70
    )

    print(
        "\nOnly held-out validation predictions will be generated."
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

        print(
            "\n" + "-" * 70
        )

        print(
            f"MODEL {class_number} OF {n_classes}"
        )

        print(
            f"Target class: {class_label}"
        )

        print(
            "-" * 70
        )


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
            f"Positive training observations: {positive_count:,}"
        )

        print(
            f"Negative training observations: {negative_count:,}"
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


        print(
            f"Model random seed:      {model_seed}"
        )

        print(
            f"Prediction random seed: {prediction_seed}"
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
            # BART latent regression function.
            #
            # BART output is real-valued and is therefore NOT
            # used directly as a Bernoulli probability.
            # ------------------------------------------------

            mu = pmb.BART(
                "mu",
                X=X_data,
                Y=y_train_binary,
                m=BART_TREES,
            )


            # ------------------------------------------------
            # Logistic link.
            #
            # This maps the real-valued BART output to (0, 1).
            # ------------------------------------------------

            p = pm.Deterministic(
                "p",
                pm.math.sigmoid(
                    mu
                ),
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
            # SWITCH TO VALIDATION DATA
            # =================================================

            print(
                "\nSwitching model to held-out validation data...",
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
                "Generating held-out validation predictions...",
                flush=True,
            )


            prediction_idata = pm.sample_posterior_predictive(
                trace,
                var_names=["p"],
                predictions=True,
                random_seed=prediction_seed,
                progressbar=True,
                return_inferencedata=True,
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
        # IDENTIFY OBSERVATION DIMENSION
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

        print(
            "\nCalculating posterior mean validation probabilities...",
            flush=True,
        )


        mean_probability = (
            p_samples
            .mean(
                dim=[
                    "chain",
                    "draw",
                ]
            )
        )


        test_probability = np.asarray(
            mean_probability.values,
            dtype=np.float64,
        )


        # ----------------------------------------------------
        # The posterior mean should now contain only the
        # validation observation dimension.
        # ----------------------------------------------------

        test_probability = np.squeeze(
            test_probability
        )


        # ====================================================
        # VALIDATE PREDICTION VECTOR
        # ====================================================

        if test_probability.ndim != 1:

            raise ValueError(
                "BART validation probabilities did not reduce "
                "to a one-dimensional vector.\n"
                f"Prediction dimensions: {p_samples.dims}\n"
                f"Prediction shape: {p_samples.shape}\n"
                f"Reduced shape: {test_probability.shape}\n"
                f"Expected: ({len(y_test)},)"
            )


        if test_probability.shape[0] != len(y_test):

            raise ValueError(
                "Unexpected number of BART validation predictions.\n"
                f"Class: {class_label}\n"
                f"Expected: {len(y_test)}\n"
                f"Received: {test_probability.shape[0]}"
            )


        # ====================================================
        # NUMERICAL SAFETY
        # ====================================================

        test_probability = np.nan_to_num(
            test_probability,
            nan=0.5,
            posinf=1.0,
            neginf=0.0,
        )


        test_probability = np.clip(
            test_probability,
            0.0,
            1.0,
        )


        # ====================================================
        # STORE CLASS PROBABILITIES
        # ====================================================

        bart_probability_matrix[
            :,
            class_number - 1,
        ] = test_probability


        print(
            f"Stored {len(test_probability):,} "
            f"validation probabilities for '{class_label}'.",
            flush=True,
        )


        # ====================================================
        # MODEL SUMMARY
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

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL ONE-VS-REST BART MODELS COMPLETED"
    )

    print(
        "=" * 70
    )


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


    if (
        probability_sum <= 1e-12
    ).any():

        raise ValueError(
            "At least one validation observation has a "
            "near-zero total class probability."
        )


    bart_probability_matrix = (
        bart_probability_matrix
        / probability_sum
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
            "Classification_Link": ["Sigmoid"],
            "Approach": ["One-vs-rest"],
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
    # CONFIGURATION TABLE
    # ========================================================

    configuration_df = pd.DataFrame(
        {
            "Parameter": [
                "Model",
                "Approach",
                "Target",
                "Training_Observations",
                "Validation_Observations",
                "Number_of_Classes",
                "Trees_Per_BART_Model",
                "Posterior_Draws",
                "Tuning_Draws",
                "Chains",
                "Cores",
                "Validation_Size",
                "Random_State",
                "Random_Seed",
                "Categorical_Encoding",
                "Identifier_Used_For_Modeling",
                "Classification_Link",
            ],
            "Value": [
                "BART",
                "One-vs-rest binary BART",
                TARGET,
                len(y_train),
                len(y_test),
                n_classes,
                BART_TREES,
                BART_DRAWS,
                BART_TUNE,
                BART_CHAINS,
                BART_CORES,
                TEST_SIZE,
                RANDOM_STATE,
                RANDOM_SEED,
                "One-hot encoding",
                "No",
                "Sigmoid",
            ],
        }
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "SAVING BART RESULTS"
    )

    print(
        "=" * 70
    )


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


    configuration_df.to_csv(
        os.path.join(
            RESULT_DIR,
            "bart_configuration.csv",
        ),
        index=False,
    )


    # ========================================================
    # CONSOLE RESULTS
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "BART RESULTS"
    )

    print(
        "=" * 70
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
        f"Training observations:   {len(y_train):,}"
    )


    print(
        f"Validation observations: {len(y_test):,}"
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

    print(
        "\n" + "=" * 70
    )

    print(
        "BART RESULT ARTIFACTS"
    )

    print(
        "=" * 70
    )


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
            "14_bart_validation_andevaluation.py "
            "completed successfully.\n"
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
          f"Validation_Observations: {len(y_test)}\n"
        )
        
        marker.write(
          f"Number_of_Classes: {n_classes}\n"
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
          f"Random_State: {RANDOM_STATE}\n"
        )

        marker.write(
          f"Random_Seed: {RANDOM_SEED}\n"
        )
        
        marker.write(
          "Classification_Link: Sigmoid\n"
        )
        
        marker.write(
          "Approach: One-vs-rest\n"
        )
        
        print(
          "14_bart_validation_andevaluation.py "
          "COMPLETED SUCCESSFULLY"
        )
        
        print(
          "=" * 70
        )
        
        print(
          f"\nResults saved to:\n{RESULT_DIR}"
        )
        
# ============================================================
# WINDOWS MULTIPROCESSING ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()












