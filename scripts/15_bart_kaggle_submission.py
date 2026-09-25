#######################################################
# 15_bart_kaggle_submission.py
#
# FINAL BART KAGGLE SUBMISSION
#
# Assignment 6
#
# Purpose
# -------
# Fit seven one-vs-rest BART binary classifiers using ALL labeled training
# observations, generate:
#
# 1. Raw / pre-normalization BART probabilities for the Kaggle test set.
# 2. Raw / pre-normalization BART probabilities for the training set.
# 3. BART training predictions.
# 4. Normalized seven-class probabilities for Kaggle submission.
# 5. Final Kaggle submission.
#
# IMPORTANT
# ---------
# BART's latent output is on the real line.
#
# For binary classification we therefore use:
#
#    mu = BART(...)
#    p  = sigmoid(mu)
#    y  ~ Bernoulli(p)
#
# DO NOT pass `mu` directly as `p`.
#
# This script intentionally uses:
#    BART trees:      10
#    Posterior draws: 100
#    Tuning draws:    100
#     Chains:          1
#     Cores:           1
#
# These are the settings currently being tested in the project.
#
# Outputs
# -------
# The script writes files into:
#
#    output/bart/
#
# including:
#
#    bart_raw_test_probabilities.csv
#    bart_raw_train_probabilities.csv
#    bart_train_predictions.csv
#     bart_normalized_test_probabilities.csv
#     submission_bart.csv
#
# It also writes one per-class diagnostic file for transparency.
#########################################################

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import pymc as pm
import pymc_bart as pmb
import multiprocessing as mp

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 8555

BART_TREES = 10
POSTERIOR_DRAWS = 100
TUNE_DRAWS = 100

CHAINS = 1
CORES = 1

TARGET = "NObeyesdad"
ID_COLUMN = "id"

# Keep class order explicit and stable.
CLASS_NAMES = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
    "Overweight_Level_I",
    "Overweight_Level_II",
]

N_CLASSES = len(CLASS_NAMES)


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

# Search several likely locations.
DATA_CANDIDATES = [
    PROJECT_DIR / "data",
    PROJECT_DIR / "Data",
    PROJECT_DIR / "datasets",
    PROJECT_DIR / "dataset",
    PROJECT_DIR,
]

OUTPUT_DIR = PROJECT_DIR / "output" / "bart"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def find_file(filename: str) -> Path:
    ###############################################
    # Search common project locations for a filename.
    ###############################################
    candidates = []

    for directory in DATA_CANDIDATES:
        candidates.append(directory / filename)

    # Also recursively search the project if the direct paths fail.
    for path in candidates:
        if path.exists():
            return path

    recursive_matches = list(PROJECT_DIR.rglob(filename))

    if recursive_matches:
        # Prefer files that are not inside outputs.
        recursive_matches = [
            p for p in recursive_matches
            if "outputs" not in p.parts
        ]

        if recursive_matches:
            return recursive_matches[0]

    raise FileNotFoundError(
        f"\nCould not find '{filename}'.\n"
        f"Project directory searched: {PROJECT_DIR}\n"
        f"Please make sure train.csv, test.csv, and sample_submission.csv "
        f"are present in the project."
    )


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    ##########################################################
    # Numerically stable sigmoid for NumPy arrays.
    ##########################################################
    x = np.asarray(x, dtype=np.float64)

    result = np.empty_like(x)

    positive = x >= 0
    negative = ~positive

    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    exp_x = np.exp(x[negative])
    result[negative] = exp_x / (1.0 + exp_x)

    return result


def extract_posterior_variable(
    idata,
    variable_name: str,
) -> np.ndarray:
    ######################################################################
    # Extract an ArviZ posterior variable and flatten chain/draw dimensions.
    #
    # Expected BART output shape:
    #
    #    (chain, draw, observation)
    #
    # Returned shape:
    #
    #    (posterior_sample, observation)
    ######################################################################

    if not hasattr(idata, "posterior"):
        raise RuntimeError("InferenceData does not contain a posterior group.")

    if variable_name not in idata.posterior:
        available = list(idata.posterior.data_vars)
        raise KeyError(
            f"Posterior variable '{variable_name}' not found. "
            f"Available variables: {available}"
        )

    values = idata.posterior[variable_name].values

    if values.ndim < 3:
        raise RuntimeError(
            f"Unexpected posterior shape for '{variable_name}': "
            f"{values.shape}"
        )

    # Combine chain and draw dimensions.
    values = values.reshape(
        -1,
        values.shape[-1],
    )

    return values


def posterior_probability_from_mu(
    mu_samples: np.ndarray,
) -> np.ndarray:
    ###############################################################
    # Convert raw BART latent values to Bernoulli probabilities.
    #
    # Input:
    #    samples x observations
    #
    # Output:
    #    samples x observations
    ###############################################################
    return sigmoid_np(mu_samples)


def summarize_probabilities(
    probabilities: np.ndarray,
    class_name: str,
    label: str,
) -> None:
    ###############################################################
    # Print useful diagnostics for posterior probability samples.
    #
    # probabilities:
    #    posterior_samples x observations
    ###############################################################

    flat = probabilities.reshape(-1)

    print()
    print(f"{class_name} - {label}")
    print("-" * 70)
    print(f"Posterior samples: {probabilities.shape[0]:,}")
    print(f"Observations:      {probabilities.shape[1]:,}")
    print(f"Mean:              {flat.mean():.6f}")
    print(f"Std Dev:           {flat.std():.6f}")
    print(f"Variance:          {flat.var():.6f}")
    print(f"Min:               {flat.min():.6f}")
    print(f"25%:               {np.quantile(flat, 0.25):.6f}")
    print(f"Median:            {np.median(flat):.6f}")
    print(f"75%:               {np.quantile(flat, 0.75):.6f}")
    print(f"Max:               {flat.max():.6f}")


def posterior_mean(
    probability_samples: np.ndarray,
) -> np.ndarray:
    #####################################################
    # Posterior mean probability for every observation.
    #####################################################
    return probability_samples.mean(axis=0)


def posterior_mode_prediction(
    probability_samples: np.ndarray,
) -> np.ndarray:
    #############################################################
    # Convert posterior probabilities into binary BART predictions.
    #
    # Threshold is 0.5.
    #############################################################
    return (probability_samples.mean(axis=0) >= 0.5).astype(int)


def save_class_diagnostics(
    class_name: str,
    train_probabilities: np.ndarray,
    test_probabilities: np.ndarray,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
) -> None:
    ############################################################
    # Save posterior mean/std probabilities for one class.
    ############################################################

    train_mean = train_probabilities.mean(axis=0)
    train_std = train_probabilities.std(axis=0)

    test_mean = test_probabilities.mean(axis=0)
    test_std = test_probabilities.std(axis=0)

    safe_name = class_name.replace("/", "_").replace(" ", "_")

    train_df = pd.DataFrame(
        {
            ID_COLUMN: train_ids,
            "Probability_Mean": train_mean,
            "Probability_Std": train_std,
            "BART_Prediction": (train_mean >= 0.5).astype(int),
        }
    )

    test_df = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            "Probability_Mean": test_mean,
            "Probability_Std": test_std,
        }
    )

    train_df.to_csv(
        OUTPUT_DIR / f"bart_{safe_name}_train_diagnostics.csv",
        index=False,
    )

    test_df.to_csv(
        OUTPUT_DIR / f"bart_{safe_name}_test_diagnostics.csv",
        index=False,
    )


def check_probability_matrix(
    probability_df: pd.DataFrame,
    name: str,
) -> None:
    ############################################
    # Verify probabilities are numerically valid.
    ############################################

    values = probability_df.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise RuntimeError(
            f"{name} contains NaN or infinite probabilities."
        )

    if (values < 0).any() or (values > 1).any():
        raise RuntimeError(
            f"{name} contains probabilities outside [0, 1]."
        )


def normalize_class_probabilities(
    raw_probabilities: pd.DataFrame,
) -> pd.DataFrame:
    ##################################################################
    # Normalize independent one-vs-rest probabilities so that the seven
    # class probabilities sum to exactly 1 for every observation.
    #
    # This is NOT the raw BART output.
    #
    # Raw one-vs-rest BART probabilities are retained separately.
    #
    # The normalized values are:
    #
    #    p_k / sum_j p_j
    ##################################################################

    values = raw_probabilities.to_numpy(dtype=np.float64)

    row_sums = values.sum(axis=1)

    if not np.isfinite(row_sums).all():
        raise RuntimeError(
            "Probability normalization encountered NaN/Inf row sums."
        )

    if (row_sums <= 0).any():
        bad = np.where(row_sums <= 0)[0][:10]
        raise RuntimeError(
            f"Cannot normalize rows with non-positive probability sums. "
            f"Example rows: {bad.tolist()}"
        )

    normalized = values / row_sums[:, None]

    normalized_df = pd.DataFrame(
        normalized,
        columns=raw_probabilities.columns,
    )

    return normalized_df


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    warnings.filterwarnings("ignore", category=FutureWarning)

    banner("BART KAGGLE SUBMISSION")

    print("ASSIGNMENT 6")
    print()
    print("This script fits final BART models using ALL labeled")
    print("training observations.")
    print("No validation split is used.")
    print()
    print("Training-set posterior predictions and raw Kaggle")
    print("probabilities will be saved.")

    print()
    print("Configuration:")
    print(f"Random seed:       {RANDOM_SEED}")
    print(f"BART trees:        {BART_TREES}")
    print(f"Posterior draws:   {POSTERIOR_DRAWS}")
    print(f"Tuning draws:      {TUNE_DRAWS}")
    print(f"Chains:            {CHAINS}")
    print(f"Cores:             {CORES}")

    # ----------------------------------------------------------------------
    # ENVIRONMENT
    # ----------------------------------------------------------------------

    banner("PYTHON ENVIRONMENT")

    print(f"Python:     {sys.version}")
    print(f"Executable: {sys.executable}")
    print(f"NumPy:      {np.__version__}")
    print(f"Pandas:     {pd.__version__}")
    print(f"PyMC:       {pm.__version__}")
    print(f"PyMC-BART:  {pmb.__version__}")

    # ----------------------------------------------------------------------
    # LOCATE DATA
    # ----------------------------------------------------------------------

    banner("LOADING DATA")

    train_path = find_file("train.csv")
    test_path = find_file("test.csv")
    sample_path = find_file("sample_submission.csv")

    print(f"Training file:        {train_path}")
    print(f"Kaggle test file:     {test_path}")
    print(f"Sample submission:    {sample_path}")

    print()
    print("Loading training data...")
    train = pd.read_csv(train_path)

    print("Training data loaded.")
    print(f"Training rows:    {len(train):,}")
    print(f"Training columns: {len(train.columns):,}")

    print()
    print("Loading Kaggle test data...")
    test = pd.read_csv(test_path)

    print("Test data loaded.")
    print(f"Test rows:    {len(test):,}")
    print(f"Test columns: {len(test.columns):,}")

    print()
    print("Loading sample submission...")
    sample_submission = pd.read_csv(sample_path)

    print("Sample submission loaded.")
    print(f"Sample rows:    {len(sample_submission):,}")
    print(f"Sample columns: {len(sample_submission.columns):,}")

    # ----------------------------------------------------------------------
    # VALIDATE DATA
    # ----------------------------------------------------------------------

    banner("DATA VALIDATION")

    required_train_columns = {
        ID_COLUMN,
        TARGET,
    }

    missing_train = required_train_columns - set(train.columns)

    if missing_train:
        raise RuntimeError(
            f"Required training columns are missing: "
            f"{sorted(missing_train)}"
        )

    if ID_COLUMN not in test.columns:
        raise RuntimeError(
            "The 'id' column was not found in test.csv."
        )

    if TARGET in test.columns:
        raise RuntimeError(
            "test.csv unexpectedly contains the target column "
            f"'{TARGET}'."
        )

    if ID_COLUMN not in sample_submission.columns:
        raise RuntimeError(
            "The 'id' column was not found in sample_submission.csv."
        )

    if TARGET not in sample_submission.columns:
        raise RuntimeError(
            f"The target column '{TARGET}' was not found in "
            "sample_submission.csv."
        )

    if len(sample_submission) != len(test):
        raise RuntimeError(
            "sample_submission.csv and test.csv have different row counts: "
            f"{len(sample_submission)} vs {len(test)}"
        )

    actual_classes = sorted(train[TARGET].dropna().unique().tolist())
    expected_classes = sorted(CLASS_NAMES)

    print("Classes found in training data:")
    for cls in actual_classes:
        print(f"  {cls}")

    if actual_classes != expected_classes:
        raise RuntimeError(
            "\nUnexpected target classes.\n"
            f"Expected: {expected_classes}\n"
            f"Found:    {actual_classes}"
        )

    if train[TARGET].isna().any():
        raise RuntimeError(
            "Training target contains missing values."
        )

    # ----------------------------------------------------------------------
    # TARGET DISTRIBUTION
    # ----------------------------------------------------------------------

    banner("TARGET DISTRIBUTION")

    target_counts = train[TARGET].value_counts()

    for class_name in CLASS_NAMES:
        print(
            f"{class_name:25s} "
            f"{int(target_counts[class_name]):,}"
        )

    # ----------------------------------------------------------------------
    # IDENTIFIERS
    # ----------------------------------------------------------------------

    train_ids = train[ID_COLUMN].to_numpy()
    test_ids = test[ID_COLUMN].to_numpy()

    print()
    print("Training ID range:")
    print(f"{train_ids.min()} {train_ids.max()}")

    print()
    print("Test ID range:")
    print(f"{test_ids.min()} {test_ids.max()}")

    # ----------------------------------------------------------------------
    # FEATURES
    # ----------------------------------------------------------------------

    feature_columns = [
        col
        for col in train.columns
        if col not in [ID_COLUMN, TARGET]
    ]

    missing_test_features = [
        col
        for col in feature_columns
        if col not in test.columns
    ]

    if missing_test_features:
        raise RuntimeError(
            "The following training features are missing from test.csv: "
            f"{missing_test_features}"
        )

    extra_test_columns = [
        col
        for col in test.columns
        if col not in feature_columns + [ID_COLUMN]
    ]

    if extra_test_columns:
        print()
        print("WARNING: Extra test columns detected:")
        for col in extra_test_columns:
            print(f"  {col}")

    print()
    print("Predictor columns:")
    for col in feature_columns:
        print(f"  {col}")

    print()
    print(f"Number of predictors before encoding: {len(feature_columns)}")

    # ----------------------------------------------------------------------
    # ENCODING
    # ----------------------------------------------------------------------

    banner("ENCODING PREDICTORS")

    print("Encoding categorical predictors...")

    # Combine train and test before one-hot encoding so the exact same
    # columns are generated for both datasets.
    combined = pd.concat(
        [
            train[feature_columns],
            test[feature_columns],
        ],
        axis=0,
        ignore_index=True,
    )

    categorical_columns = combined.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numerical_columns = [
        col
        for col in feature_columns
        if col not in categorical_columns
    ]

    print()
    print("Categorical columns:")
    for col in categorical_columns:
        print(f"  {col}")

    print()
    print("Numeric columns:")
    for col in numerical_columns:
        print(f"  {col}")

    # Convert categoricals to strings so missing values and categories are
    # handled consistently.
    for col in categorical_columns:
        combined[col] = (
            combined[col]
            .astype("string")
            .fillna("__MISSING__")
        )

    # Numeric columns: force numeric and fill missing values with the
    # training median.
    for col in numerical_columns:
        combined[col] = pd.to_numeric(
            combined[col],
            errors="coerce",
        )

        train_part = combined.iloc[: len(train)]

        median_value = train_part[col].median()

        if pd.isna(median_value):
            median_value = 0.0

        combined[col] = combined[col].fillna(median_value)

    # One-hot encode categorical predictors.
    combined_encoded = pd.get_dummies(
        combined,
        columns=categorical_columns,
        drop_first=False,
        dtype=float,
    )

    # Final safety conversion.
    combined_encoded = combined_encoded.astype(np.float64)

    X_train = combined_encoded.iloc[: len(train)].to_numpy(
        dtype=np.float64
    )

    X_test = combined_encoded.iloc[len(train):].to_numpy(
        dtype=np.float64
    )

    print()
    print("Encoded training matrix:")
    print(f"Rows:    {X_train.shape[0]:,}")
    print(f"Columns: {X_train.shape[1]:,}")

    print()
    print("Encoded Kaggle test matrix:")
    print(f"Rows:    {X_test.shape[0]:,}")
    print(f"Columns: {X_test.shape[1]:,}")

    if X_train.shape[1] != X_test.shape[1]:
        raise RuntimeError(
            "Encoded training/test matrices have different numbers "
            "of columns."
        )

    if not np.isfinite(X_train).all():
        raise RuntimeError(
            "X_train contains NaN or infinite values after encoding."
        )

    if not np.isfinite(X_test).all():
        raise RuntimeError(
            "X_test contains NaN or infinite values after encoding."
        )

    # ----------------------------------------------------------------------
    # TARGET ENCODING
    # ----------------------------------------------------------------------

    y_full = train[TARGET].to_numpy()

    # ----------------------------------------------------------------------
    # STORAGE
    # ----------------------------------------------------------------------

    raw_train_probabilities = pd.DataFrame(
        index=np.arange(len(train))
    )

    raw_test_probabilities = pd.DataFrame(
        index=np.arange(len(test))
    )

    train_binary_predictions = pd.DataFrame(
        {
            ID_COLUMN: train_ids
        }
    )

    # Store posterior probability standard deviations as well.
    train_probability_sd = pd.DataFrame(
        {
            ID_COLUMN: train_ids
        }
    )

    test_probability_sd = pd.DataFrame(
        {
            ID_COLUMN: test_ids
        }
    )

    # ----------------------------------------------------------------------
    # FIT SEVEN ONE-VS-REST MODELS
    # ----------------------------------------------------------------------

    banner("FITTING FINAL ONE-VS-REST BART MODELS")

    print()
    print("Each BART model uses ALL labeled training observations.")
    print()
    print("IMPORTANT MODEL FORM:")
    print()
    print("    mu = BART(X, Y)")
    print("    p  = sigmoid(mu)")
    print("    y  ~ Bernoulli(p)")
    print()
    print("The sigmoid is essential because BART's latent output")
    print("can range over the real line.")

    for class_index, class_name in enumerate(CLASS_NAMES):

        banner(
            f"MODEL {class_index + 1} OF {N_CLASSES}: {class_name}"
        )

        y_binary = (
            train[TARGET].to_numpy() == class_name
        ).astype(np.int8)

        positive_count = int(y_binary.sum())
        negative_count = int(len(y_binary) - positive_count)

        print(f"Positive observations: {positive_count:,}")
        print(f"Negative observations: {negative_count:,}")

        model_seed = RANDOM_SEED + class_index + 1
        prediction_seed = RANDOM_SEED + 101 + class_index

        print(f"Model random seed:      {model_seed}")
        print(f"Prediction random seed: {prediction_seed}")


        #########################################################################
                # ------------------------------------------------------------------
        # BUILD MODEL
        # ------------------------------------------------------------------

        print()
        print("Building BART model...")

        with pm.Model() as model:

            # Mutable predictor container.
            #
            # The BART model is fitted on X_train and can subsequently
            # be evaluated on X_test by replacing this data container.
            X_data = pm.Data(
                "X_data",
                X_train,
            )

            # Keep the training response fixed.
            #
            # IMPORTANT:
            # y_binary is the actual response used by BART during fitting.
            # We do NOT replace it with fake test responses later.
            mu = pmb.BART(
                "mu",
                X=X_data,
                Y=y_binary,
                m=BART_TREES,
            )

            # Binary-classification probability.
            #
            # BART's latent output is on the real line, so convert it
            # explicitly to [0, 1].
            p = pm.Deterministic(
                "p",
                pm.math.sigmoid(mu),
            )

            # Bernoulli likelihood for the training observations.
            y_obs = pm.Bernoulli(
                "y_obs",
                p=p,
                observed=y_binary,
            )

            # --------------------------------------------------------------
            # SAMPLE POSTERIOR
            # --------------------------------------------------------------

            print()
            print("Starting posterior sampling...")

            trace = pm.sample(
                draws=POSTERIOR_DRAWS,
                tune=TUNE_DRAWS,
                chains=CHAINS,
                cores=CORES,
                random_seed=model_seed,
                progressbar=True,
                compute_convergence_checks=False,
                return_inferencedata=True,
            )

            print()
            print("Posterior sampling completed.")

            # --------------------------------------------------------------
            # TRAINING-SET PREDICTIONS
            # --------------------------------------------------------------

            print()
            print("Extracting training-set BART latent posterior...")

            # Extract the BART latent function directly.
            #
            # Expected shape:
            #
            #     chain x draw x observation
            #
            # extract_posterior_variable() converts this to:
            #
            #     posterior_sample x observation
            #
            train_mu_posterior = extract_posterior_variable(
                trace,
                "mu",
            )

            if train_mu_posterior.shape[1] != len(train):
                raise RuntimeError(
                    f"Training BART latent posterior shape is "
                    f"{train_mu_posterior.shape}, expected second dimension "
                    f"{len(train)}."
                )

            # Explicitly convert BART latent values to probabilities.
            train_posterior = posterior_probability_from_mu(
                train_mu_posterior
            )

            if not np.isfinite(train_posterior).all():
                raise RuntimeError(
                    "Training BART probabilities contain NaN or infinite values."
                )

            train_mean = posterior_mean(
                train_posterior
            )

            train_sd = train_posterior.std(
                axis=0
            )

            train_pred = (
                train_mean >= 0.5
            ).astype(int)

            # Save class probability and binary prediction.
            raw_train_probabilities[class_name] = train_mean

            train_probability_sd[class_name] = train_sd

            train_binary_predictions[class_name] = train_pred

            print()
            print("TRAINING-SET BART RESULTS")
            print(f"Mean probability:   {train_mean.mean():.6f}")
            print(f"Median probability: {np.median(train_mean):.6f}")
            print(f"Min probability:    {train_mean.min():.6f}")
            print(f"Max probability:    {train_mean.max():.6f}")
            print(
                "Predicted positive observations: "
                f"{int(train_pred.sum()):,}"
            )
##########################################################
        # --------------------------------------------------------------
        # TEST PREDICTIONS
        # --------------------------------------------------------------

        print()
        print("Switching model to Kaggle test data...")

        # IMPORTANT:
        #
        # X_data is the only mutable predictor container in this model.
        # Do NOT attempt to set y_data here because y_data is not part
        # of the current model.
        #
        # Also, model.set_data() is explicitly called on `model` rather
        # than pm.set_data(), so we do not depend on the model context
        # stack.

        model.set_data(
            "X_data",
            X_test,
        )

        print("Generating Kaggle test predictions...")

        # IMPORTANT:
        #
        # Keep posterior predictive sampling inside the model context.
        # This prevents:
        #
        #     TypeError: No model on context stack.
        #
        # from occurring.

        with model:

            prediction_idata = pm.sample_posterior_predictive(
                trace,
                var_names=["p"],
                predictions=True,
                random_seed=prediction_seed,
                progressbar=True,
                return_inferencedata=True,
            )

        print()
        print("Kaggle test BART prediction completed.")

        # --------------------------------------------------------------
        # EXTRACT TEST PROBABILITIES
        # --------------------------------------------------------------

        if hasattr(prediction_idata, "predictions"):
            prediction_group = prediction_idata.predictions

        elif hasattr(prediction_idata, "posterior_predictive"):
            prediction_group = prediction_idata.posterior_predictive

        else:
            raise RuntimeError(
                "Posterior prediction result contains neither "
                "'predictions' nor 'posterior_predictive'."
            )

        if "p" not in prediction_group:
            raise RuntimeError(
                "Predicted probability variable 'p' was not found."
            )

        # Expected shape:
        #
        #     (chain, draw, observation)
        #
        # Example:
        #
        #     (1, 100, 13840)

        test_posterior = prediction_group["p"].values

        print()
        print(
            "Raw test BART posterior probability shape:",
            test_posterior.shape,
        )

        if test_posterior.ndim < 3:
            raise RuntimeError(
                f"Unexpected test probability shape: "
                f"{test_posterior.shape}"
            )

        # Combine chain and posterior-draw dimensions.
        #
        # Before:
        #
        #     chain x draw x observation
        #
        # After:
        #
        #     posterior_sample x observation

        test_posterior = test_posterior.reshape(
            -1,
            test_posterior.shape[-1],
        )

        print(
            "Reshaped test posterior probability shape:",
            test_posterior.shape,
        )

        if test_posterior.shape[1] != len(test):
            raise RuntimeError(
                f"Test posterior probability shape is "
                f"{test_posterior.shape}, but expected "
                f"{len(test)} observations."
            )

        # --------------------------------------------------------------
        # POSTERIOR MEAN / STANDARD DEVIATION
        # --------------------------------------------------------------

        test_mean = posterior_mean(
            test_posterior
        )

        test_sd = test_posterior.std(
            axis=0
        )

        # Store the posterior mean as the raw one-vs-rest probability.
        #
        # IMPORTANT:
        #
        # These are still the raw BART binary probabilities.
        # They have NOT been normalized across the seven classes.

        raw_test_probabilities[class_name] = test_mean

        test_probability_sd[class_name] = test_sd

        print()
        print("RAW TEST BART PROBABILITIES")
        print(
            f"Mean probability:      {test_mean.mean():.6f}"
        )
        print(
            f"Median probability:    {np.median(test_mean):.6f}"
        )
        print(
            f"Min probability:       {test_mean.min():.6f}"
        )
        print(
            f"Max probability:       {test_mean.max():.6f}"
        )
        print(
            f"Std of probabilities:  {test_mean.std():.6f}"
        )

        # --------------------------------------------------------------
        # SAVE PER-CLASS DIAGNOSTICS
        # --------------------------------------------------------------

        save_class_diagnostics(
            class_name=class_name,
            train_probabilities=train_posterior,
            test_probabilities=test_posterior,
            train_ids=train_ids,
            test_ids=test_ids,
        )

        # --------------------------------------------------------------
        # CLEAN UP CURRENT MODEL
        # --------------------------------------------------------------

        del prediction_idata
        del trace

        print()
        print(f"Completed model: {class_name}")

    # =========================================================================
    # VALIDATE RAW PROBABILITIES
    # =========================================================================

    banner("RAW BART PROBABILITY VALIDATION")

    raw_train_probabilities.insert(
        0,
        ID_COLUMN,
        train_ids,
    )

    raw_test_probabilities.insert(
        0,
        ID_COLUMN,
        test_ids,
    )

    train_probability_sd.insert(
        0,
        "Row_ID",
        np.arange(len(train)),
    )

    test_probability_sd.insert(
        0,
        "Row_ID",
        np.arange(len(test)),
    )

    raw_train_probability_values = raw_train_probabilities[
        CLASS_NAMES
    ]

    raw_test_probability_values = raw_test_probabilities[
        CLASS_NAMES
    ]

    check_probability_matrix(
        raw_train_probability_values,
        "Raw training BART probabilities",
    )

    check_probability_matrix(
        raw_test_probability_values,
        "Raw test BART probabilities",
    )

    print()
    print("Raw training probability range:")
    print(
        f"min={raw_train_probability_values.min().min():.6f}, "
        f"max={raw_train_probability_values.max().max():.6f}"
    )

    print()
    print("Raw test probability range:")
    print(
        f"min={raw_test_probability_values.min().min():.6f}, "
        f"max={raw_test_probability_values.max().max():.6f}"
    )

    # =========================================================================
    # SAVE RAW PROBABILITIES
    # =========================================================================

    banner("SAVING RAW BART PROBABILITIES")

    raw_train_path = (
        OUTPUT_DIR /
        "bart_raw_train_probabilities.csv"
    )

    raw_test_path = (
        OUTPUT_DIR /
        "bart_raw_test_probabilities.csv"
    )

    train_sd_path = (
        OUTPUT_DIR /
        "bart_train_probability_sd.csv"
    )

    test_sd_path = (
        OUTPUT_DIR /
        "bart_test_probability_sd.csv"
    )

    raw_train_probabilities.to_csv(
        raw_train_path,
        index=False,
    )

    raw_test_probabilities.to_csv(
        raw_test_path,
        index=False,
    )

    train_probability_sd.to_csv(
        train_sd_path,
        index=False,
    )

    test_probability_sd.to_csv(
        test_sd_path,
        index=False,
    )

    print(f"Saved: {raw_train_path}")
    print(f"Saved: {raw_test_path}")
    print(f"Saved: {train_sd_path}")
    print(f"Saved: {test_sd_path}")

    # =========================================================================
    # TRAINING PREDICTIONS
    # =========================================================================

    banner("TRAINING-SET BART PREDICTIONS")

    train_prediction_output = train_binary_predictions.copy()

    # Keep actual target next to predictions for inspection.
    train_prediction_output.insert(
        1,
        "Actual",
        y_full,
    )

    train_prediction_path = (
        OUTPUT_DIR /
        "bart_train_predictions.csv"
    )

    train_prediction_output.to_csv(
        train_prediction_path,
        index=False,
    )

    print(
        f"Saved: {train_prediction_path}"
    )

    # -------------------------------------------------------------------------
    # Training binary prediction summary
    # -------------------------------------------------------------------------

    print()
    print("ONE-VS-REST TRAINING PREDICTION COUNTS")

    for class_name in CLASS_NAMES:

        count = int(
            train_binary_predictions[class_name].sum()
        )

        print(
            f"{class_name:25s} {count:,}"
        )

    # =========================================================================
    # RAW TEST PROBABILITY SUMMARY
    # =========================================================================

    banner("RAW TEST BART PROBABILITY SUMMARY")

    print()

    print(
        raw_test_probability_values.describe().T[
            [
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        ]
    )

    print()
    print("RAW TEST MEAN PROBABILITY")

    print(
        raw_test_probability_values.mean()
        .sort_values()
    )

    print()
    print("RAW TEST MEDIAN PROBABILITY")

    print(
        raw_test_probability_values.median()
        .sort_values()
    )

    # =========================================================================
    # NORMALIZATION
    # =========================================================================

    banner("SEVEN-CLASS PROBABILITY NORMALIZATION")

    print()
    print(
        "The raw one-vs-rest BART probabilities do NOT necessarily "
        "sum to 1."
    )

    print()
    print(
        "For the final multiclass submission they are normalized "
        "row-wise:"
    )

    print()
    print(
        "    normalized_p_k = p_k / sum(p_1 ... p_7)"
    )

    print()

    # -------------------------------------------------------------------------
    # NORMALIZE TEST PROBABILITIES
    # -------------------------------------------------------------------------

    normalized_test_probability_values = (
        normalize_class_probabilities(
            raw_test_probability_values
        )
    )

    # Preserve the Kaggle test IDs.

    normalized_test_probabilities = (
        normalized_test_probability_values.copy()
    )

    normalized_test_probabilities.insert(
        0,
        ID_COLUMN,
        test_ids,
    )

    # -------------------------------------------------------------------------
    # VALIDATE NORMALIZED PROBABILITIES
    # -------------------------------------------------------------------------

    banner("NORMALIZED PROBABILITY VALIDATION")

    normalized_values = (
        normalized_test_probabilities[
            CLASS_NAMES
        ].to_numpy(
            dtype=np.float64
        )
    )

    if not np.isfinite(
        normalized_values
    ).all():

        raise RuntimeError(
            "Normalized test probabilities contain "
            "NaN or infinite values."
        )

    if (
        normalized_values < 0
    ).any() or (
        normalized_values > 1
    ).any():

        raise RuntimeError(
            "Normalized test probabilities contain "
            "values outside [0, 1]."
        )

    normalized_row_sums = (
        normalized_values.sum(
            axis=1
        )
    )

    print()
    print(
        "Normalized probability row-sum diagnostics:"
    )

    print(
        f"Minimum row sum: {normalized_row_sums.min():.12f}"
    )

    print(
        f"Maximum row sum: {normalized_row_sums.max():.12f}"
    )

    print(
        f"Mean row sum:    {normalized_row_sums.mean():.12f}"
    )

    if not np.allclose(
        normalized_row_sums,
        1.0,
        rtol=1e-10,
        atol=1e-10,
    ):

        raise RuntimeError(
            "Normalized probabilities do not sum to 1 "
            "within tolerance."
        )

    print()
    print(
        "SUCCESS: Every normalized test probability "
        "row sums to 1."
    )

    # -------------------------------------------------------------------------
    # SAVE NORMALIZED PROBABILITIES
    # -------------------------------------------------------------------------

    banner("SAVING NORMALIZED TEST PROBABILITIES")

    normalized_test_path = (
        OUTPUT_DIR /
        "bart_normalized_test_probabilities.csv"
    )

    normalized_test_probabilities.to_csv(
        normalized_test_path,
        index=False,
    )

    print(
        f"Saved: {normalized_test_path}"
    )

    # =========================================================================
    # FINAL MULTICLASS PREDICTIONS
    # =========================================================================

    banner("FINAL BART MULTICLASS PREDICTIONS")

    # For each Kaggle test observation, select the class with the
    # largest normalized probability.

    predicted_class_indices = np.argmax(
        normalized_values,
        axis=1,
    )

    predicted_classes = np.array(
        CLASS_NAMES,
        dtype=object,
    )[
        predicted_class_indices
    ]

    # Maximum predicted probability for diagnostics.

    predicted_probabilities = (
        normalized_values[
            np.arange(
                len(normalized_values)
            ),
            predicted_class_indices,
        ]
    )

    final_predictions = pd.DataFrame(
        {
            ID_COLUMN: test_ids,
            TARGET: predicted_classes,
            "Predicted_Probability": predicted_probabilities,
        }
    )

    print()
    print(
        "Final multiclass prediction counts:"
    )

    prediction_counts = (
        final_predictions[
            TARGET
        ]
        .value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0,
        )
    )

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:25s} "
            f"{int(prediction_counts[class_name]):,}"
        )

    print()

    print(
        "Mean maximum predicted probability: "
        f"{predicted_probabilities.mean():.6f}"
    )

    print(
        "Median maximum predicted probability: "
        f"{np.median(predicted_probabilities):.6f}"
    )

    print(
        "Minimum maximum predicted probability: "
        f"{predicted_probabilities.min():.6f}"
    )

    print(
        "Maximum maximum predicted probability: "
        f"{predicted_probabilities.max():.6f}"
    )

    # =========================================================================
    # BUILD KAGGLE SUBMISSION
    # =========================================================================

    banner("BUILDING KAGGLE SUBMISSION")

    # Start from the sample submission structure.

    submission = sample_submission.copy()

    if len(submission) != len(
        final_predictions
    ):

        raise RuntimeError(
            "Submission and prediction row counts differ: "
            f"{len(submission)} vs "
            f"{len(final_predictions)}"
        )

    # Verify expected columns.

    expected_submission_columns = [
        ID_COLUMN,
        TARGET,
    ]

    if list(
        submission.columns
    ) != expected_submission_columns:

        print()

        print(
            "WARNING: sample_submission.csv columns are "
            f"{list(submission.columns)}"
        )

        print(
            "Expected columns are "
            f"{expected_submission_columns}"
        )

    # Use the sample submission IDs as the authoritative
    # submission structure.

    if not np.array_equal(
        submission[ID_COLUMN].to_numpy(),
        test_ids,
    ):

        raise RuntimeError(
            "The IDs in sample_submission.csv do not "
            "match the IDs in test.csv in the same row order."
        )

    submission[TARGET] = predicted_classes

    # Keep only the Kaggle submission columns.

    submission = submission[
        [
            ID_COLUMN,
            TARGET,
        ]
    ]

    # =========================================================================
    # FINAL SUBMISSION VALIDATION
    # =========================================================================

    banner("FINAL SUBMISSION VALIDATION")

    if len(submission) != len(test):

        raise RuntimeError(
            "Final submission does not contain the same "
            "number of rows as test.csv: "
            f"{len(submission)} vs {len(test)}"
        )

    if submission[
        ID_COLUMN
    ].isna().any():

        raise RuntimeError(
            "Final submission contains missing IDs."
        )

    if submission[
        TARGET
    ].isna().any():

        raise RuntimeError(
            "Final submission contains missing target predictions."
        )

    submission_classes = sorted(
        submission[
            TARGET
        ].unique().tolist()
    )

    if not set(
        submission_classes
    ).issubset(
        set(CLASS_NAMES)
    ):

        raise RuntimeError(
            "Final submission contains unexpected "
            f"target classes: {submission_classes}"
        )

    if not np.array_equal(
        submission[
            ID_COLUMN
        ].to_numpy(),
        test_ids,
    ):

        raise RuntimeError(
            "Final submission IDs do not match test.csv."
        )

    print()
    print(
        "Final submission shape:"
    )

    print(
        f"Rows:    {submission.shape[0]:,}"
    )

    print(
        f"Columns: {submission.shape[1]:,}"
    )

    print()
    print(
        "Final submission columns:"
    )

    for column in submission.columns:

        print(
            f"  {column}"
        )

    print()
    print(
        "Final submission target counts:"
    )

    final_target_counts = (
        submission[
            TARGET
        ]
        .value_counts()
        .reindex(
            CLASS_NAMES,
            fill_value=0,
        )
    )

    for class_name in CLASS_NAMES:

        print(
            f"{class_name:25s} "
            f"{int(final_target_counts[class_name]):,}"
        )

    # =========================================================================
    # SAVE FINAL SUBMISSION
    # =========================================================================

    banner("SAVING FINAL KAGGLE SUBMISSION")

    submission_path = (
        OUTPUT_DIR /
        "submission_bart.csv"
    )

    submission.to_csv(
        submission_path,
        index=False,
    )

    print()

    print(
        f"Saved: {submission_path}"
    )

    # =========================================================================
    # FINAL PREVIEW
    # =========================================================================

    banner("FINAL SUBMISSION PREVIEW")

    print()

    print(
        submission.head(10).to_string(
            index=False
        )
    )

    print()
    print(
        "Last 10 submission rows:"
    )

    print(
        submission.tail(10).to_string(
            index=False
        )
    )

    # =========================================================================
    # FINAL FILE SUMMARY
    # =========================================================================

    banner("BART OUTPUT FILE SUMMARY")

    output_files = [
        raw_train_path,
        raw_test_path,
        train_sd_path,
        test_sd_path,
        train_prediction_path,
        normalized_test_path,
        submission_path,
    ]

    for path in output_files:

        if path.exists():

            size_kb = (
                path.stat().st_size
                / 1024.0
            )

            print(
                f"{path.name:45s} "
                f"{size_kb:,.1f} KB"
            )

        else:

            print(
                f"WARNING - missing: {path}"
            )

    # =========================================================================
    # COMPLETION
    # =========================================================================

    banner("BART KAGGLE SUBMISSION COMPLETE")

    print()

    print(
        "Seven one-vs-rest BART models were fitted successfully."
    )

    print(
        "Raw training probabilities were saved."
    )

    print(
        "Raw Kaggle test probabilities were saved."
    )

    print(
        "Training predictions were saved."
    )

    print(
        "Seven-class normalized probabilities were saved."
    )

    print(
        "The final multiclass Kaggle submission was saved."
    )

    print()

    print(
        "Final submission file:"
    )

    print(
        f"    {submission_path}"
    )

    print()

    print(
        "The submission is ready for upload to Kaggle."
    )

    print()


# ============================================================
# WINDOWS MULTIPROCESSING ENTRY POINT
# ============================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()
