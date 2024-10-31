"""
This module includes functions to preprocess data, configure machine
learning pipelines, perform hyperparameter tuning with grid search,
and evaluate model predictions.
"""

import json
from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from edo_case_study.modeling.util import (
    BEST_MODEL_TEST_SET_PREDICTIONS,
    display_dfs_side_by_side,
    is_outlier,
)


TEST_SIZE = 1/10
RNG = np.random.default_rng(seed=42)
HYPERPARAMETER_GRID = {
    KNeighborsRegressor.__name__: {
        "model": [KNeighborsRegressor()],
        "model__n_neighbors": [3, 5, 7, 10],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    Lasso.__name__: {
        "model": [Lasso()],
        "model__fit_intercept": [True, False],
        "model__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
    },
    LinearRegression.__name__: {
        "model": [LinearRegression()],
        "model__fit_intercept": [True, False],
    },
    RandomForestRegressor.__name__: {
        "model": [
            RandomForestRegressor(random_state=RNG.integers(0, 1e6))
        ],
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 10],
    },
    SVR.__name__: {
        "model": [SVR()],
        "model__kernel": ["linear", "rbf"],
        "model__C": [0.1, 1, 10, 100],
        "model__epsilon": [0.01, 0.1, 0.2, 0.5],
    },
    XGBRegressor.__name__: {
        "model": [
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RNG.integers(0, 1e6),
            ),
        ],
        "model__n_estimators": [50, 100, 150, 200, 250],
        "model__max_depth": [3, 6, 9, 12, 15],
        "model__learning_rate": [0.01, 0.05, 0.1, 0.3],
        "model__objective": [
            "reg:squarederror",
            "reg:pseudohubererror",
            "count:poisson",
            "reg:tweedie",
        ],
    },
}


def preprocess(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[
        np.ndarray[np.float32],
        np.ndarray[np.float32],
        np.ndarray[np.float32],
        np.ndarray[np.float32],
]:
    """
    Preprocess data by separating features and target labels, applying
    log transformation on target.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training data with features and target ``y_true`` column
    test_df : pd.DataFrame
        Testing data with features and target ``y_true`` column

    Returns
    -------
    tuple
        Processed arrays (X_train, y_train, X_test, y_test) for
        training and testing
    """

    X_train = train_df.drop(columns="y_true").values
    print(f"Train features shape: {X_train.shape}")
    y_train = np.log1p(train_df.y_true.values)
    print(f"Train labels shape: {y_train.shape}")
    X_test = test_df.drop(columns="y_true").values
    print(f"Test features shape: {X_test.shape}")
    y_test = np.log1p(test_df.y_true.values)
    print(f"Test labels shape: {y_test.shape}")
    return X_train, y_train, X_test, y_test


def configure_pipelines(
    hyperparameter_grid: dict[str, Any] = HYPERPARAMETER_GRID,
) -> list[Pipeline]:
    """
    Configure model pipelines with specified hyperparameters for grid
    search.

    Parameters
    ----------
    hyperparameter_grid : dict
        Dictionary mapping model names to their respective
        hyperparameters

    Returns
    -------
    list[Pipeline]
        List of model pipelines for each model type
    """

    pipelines = []
    for model_name, hyperparameters in hyperparameter_grid.items():
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", *hyperparameters["model"]),
            ]
        )
        pipelines.append((model_name, pipeline, hyperparameters))
    return pipelines


def get_best_model_test_set_predictions(
    best_models: dict[str, Any],
    X_test: np.ndarray[np.float32],
    y_test: np.ndarray[np.float32],
) -> BEST_MODEL_TEST_SET_PREDICTIONS:
    """
    Generate test set predictions using the best models and evaluates
    their performance.

    Parameters
    ----------
    best_models : dict
        Dictionary containing model names and their respective trained
        pipelines
    X_test : np.ndarray
        Test feature data
    y_test : np.ndarray
        True values for the test set target

    Returns
    -------
    BEST_MODEL_TEST_SET_PREDICTIONS
        Exp-transformed true test values and dictionary with
        predictions and MAE for each model
    """

    best_model_test_set_predictions = {}
    for model_name, pipeline in best_models.items():
        y_pred = pipeline.predict(X_test)
        y_pred_exp = np.expm1(y_pred)
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        best_model_test_set_predictions[model_name] = (y_pred_exp, mae)
    y_test_exp = np.expm1(y_test)
    return y_test_exp, best_model_test_set_predictions


def run_gridsearch(
    X_train: np.ndarray[np.float32],
    y_train: np.ndarray[np.float32],
    pipelines: list[Pipeline],
) -> dict[str, Any]:
    """
    Conduct grid search for each model pipeline and returns the best
    models.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature data
    y_train : np.ndarray
        Training labels
    pipelines : list
        List of model pipelines to search over

    Returns
    -------
    dict[str, Any]
        Dictionary of best-performing models found for each model type
    """

    model_header = ("=" * 50) + "\nRunning grid search for {}..."
    best_models = {}
    for model_name, pipeline, hyperparameters in pipelines:
        print(model_header.format(model_name))
        search = GridSearchCV(
            pipeline,
            param_grid=hyperparameters,
            scoring="neg_mean_absolute_error",
            cv=5,
            n_jobs=-1,
        ).fit(X_train, y_train)
        best_models[model_name] = search.best_estimator_
        best_model_hyperparameters_str = json.dumps(
            search.best_params_,
            indent=4,
            default=lambda model: type(model).__name__,
        )
        print(
            f"Best model for {model_name}:",
            best_model_hyperparameters_str,
            f"Best MAE for {model_name}: {-search.best_score_:.4f}",
            sep="\n",
        )
    return best_models


def split_data(
    features: pd.DataFrame,
    labels: pd.Series,
    rng: np.random.Generator = RNG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and testing sets, removing outliers from
    the training set.

    Parameters
    ----------
    features : pd.DataFrame
        Dataframe containing feature columns
    labels : pd.Series
        Series containing target label values
    rng : np.random.Generator, optional
        Random number generator instance, by default RNG

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Training and testing dataframes
    """

    df = pd.concat([features, labels], axis=1).dropna().astype(float)
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=rng.integers(0, 1e6)
    )
    train_df = train_df.loc[~is_outlier(train_df.y_true)]
    display_dfs_side_by_side({"train": train_df, "test": test_df})
    return train_df, test_df
