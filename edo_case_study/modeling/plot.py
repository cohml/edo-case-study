"""
This module provides a range of functions for plotting and visualizing
patterns in event data and model performance. Functions are included to
visualize event frequencies by person, subgroup, and over time (monthly
and annually).
"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas.core.groupby import DataFrameGroupBy

from edo_case_study.modeling.util import (
    BEST_MODEL_TEST_SET_PREDICTIONS,
    COLORS,
    MONTHS,
)


plt.rcParams["figure.figsize"] = (16, 8)
plt.rcParams["font.size"] = 14
plt.rcParams["grid.linestyle"] = "dashed"


def events_overall_annually(
    event_counts: pd.Series,
    event_name: str,
) -> None:
    """
    Plot distribution of events (i.e., ad exposures, site visits)
    throughout the calendar year.

    Parameters
    ----------
    event_counts : pd.Series
        Series of counts per person for the event
    event_name : str
        Name of the event being plotted

    Returns
    -------
    None
    """

    xmean, xmin, xmax = event_counts.agg(
        ["mean", "min", "max"]
    ).values
    if xmax > 10:
        xmax = ((xmax // 10) + 1) * 10
    bins = np.arange(0, xmax.item() + 2, 1)
    ax = event_counts.plot.hist(
        bins=bins,
        xlim=(xmin, xmax + 1),
        xlabel=f"Number of {event_name}",
        ylabel="Number of people",
        title=f"Number of {event_name} annually",
        alpha=1/3,
        legend=False,
    )
    ax.axvline(xmean, color="black", ls="dashed")
    ax.annotate(
        f"Mean number of {event_name}",
        (xmean, event_counts.value_counts().max()),
        rotation=90,
        va="top",
        ha="right",
    )
    ax.grid(axis="y")


def events_overall_monthly(
    events_df: pd.DataFrame,
    event_name: str,
    groupby: list[str],
) -> None:
    """
    Plot monthly distribution of events (i.e., ad exposures, site
    visits) grouped by specific variables.

    Parameters
    ----------
    events_df : pd.DataFrame
        Dataframe of event data including time and other event details
    event_name : str
        Name of the event being plotted
    groupby : list[str]
        List of column names to group data by

    Returns
    -------
    None
    """

    mean, std = (
        events_df.groupby(groupby)
                 .size()
                 .unstack()
                 .fillna(0)
                 .agg(["mean", "std"])
                 .values
    )
    xticks = list(MONTHS.keys())
    yerr_lower = mean - std
    yerr_upper = mean + std
    ax = plt.subplot()
    ax.plot(xticks, mean, lw=5)
    ax.set(
        xticks=xticks,
        xticklabels=MONTHS.values(),
        xlim=(min(xticks), max(xticks)),
        ylim=(
            yerr_lower.min() - abs(yerr_lower.min() * 3),
            yerr_upper.max() + 1,
        ),
        xlabel="Month",
        ylabel=f"Mean number of {event_name}",
        title=f"Mean number of {event_name} by month"
    )
    ax.fill_between(xticks, yerr_lower, yerr_upper, alpha=1/3)
    ax.grid(axis="x")


def events_subgroups_annually(
    events_df: pd.Series,
    event_name: str,
    groupby: list[str],
) -> None:
    """
    Plot the annual distribution of events by specified subgroups.

    Parameters
    ----------
    events_df : pd.Series
        Series of event counts per person
    event_name : str
        Name of the event being plotted
    groupby : list[str]
        Column names to group data by

    Returns
    -------
    None
    """

    groups = events_df.groupby(groupby)
    fig, axes = plt.subplots(
        groups.ngroups, sharex=True, sharey=True
    )
    for ax, (group_name, group_df) in zip(axes, groups):
        event_counts = group_df.groupby("person").size()
        xmean, xmin, xmax = event_counts.agg(
            ["mean", "min", "max"]
        ).values
        bins = np.arange(0, xmax.item() + 2, 1)
        event_counts.plot.hist(
            bins=bins,
            xlim=(xmin, min(xmax + 1, 100)),
            xlabel=f"Number of {event_name}",
            ylabel="",
            alpha=1/3,
            legend=False,
            ax=ax,
        )
        ax.axvline(xmean, color="black", ls="dashed")
        ax.set_title(f"{groupby}: {group_name}".replace("_", " "))
    if groupby == "income_bin":
        groupby = "income bin"
    elif groupby == "has_dog":
        groupby = "dog ownership"
    fig.suptitle(f"Number of {event_name} annually by {groupby}")
    fig.supylabel("Number of people")
    fig.subplots_adjust(hspace=1/2)


def events_subgroups_monthly(
    events_df: pd.DataFrame,
    event_name: str,
    groupby: str,
) -> None:
    """
    Plot monthly event distribution by subgroup based on a specified
    demographic feature.

    Parameters
    ----------
    events_df : pd.DataFrame
        Dataframe containing event data including time and group info
    event_name : str
        Name of the event being plotted
    groupby : str
        Column name for subgroup division

    Returns
    -------
    None
    """

    ax = plt.subplot()
    xticks = list(MONTHS.keys())
    title=f"Mean number of {event_name} per person by month and "
    if groupby == "income_bin":
        title += "income bin"
    elif groupby == "has_dog":
        title += "dog ownership"
    else:
        title += groupby
    for group_name, group_df in events_df.groupby(groupby):
        months = group_df.time.dt.month
        label = f"{groupby}: {group_name}".replace("_", " ")
        mean, std = (
            group_df.groupby(["person", months])
                    .size()
                    .unstack()
                    .fillna(0)
                    .agg(["mean", "std"])
                    .values
        )
        yerr_lower = mean - std
        yerr_upper = mean + std
        ax = plt.subplot()
        ax.plot(xticks, mean, lw=5, label=label)
        ax.set(
            xticks=xticks,
            xticklabels=MONTHS.values(),
            xlim=(min(xticks), max(xticks)),
            ylim=(
                yerr_lower.min() - abs(yerr_lower.min() * 1.5),
                yerr_upper.max() + 1,
            ),
            xlabel="Month",
            ylabel=f"Mean number of {event_name}",
            label=label,
            title=title,
        )
        ax.fill_between(xticks, yerr_lower, yerr_upper, alpha=1/3)
    ax.legend()
    ax.grid(axis="x")


def ads_channels_overall_annually(
    events_df: pd.DataFrame,
    event_name: str,
    groupby: str | list[str],
    title: str,
) -> None:
    """
    Plot annual distribution of ads or channels as a percentage.

    Parameters
    ----------
    events_df : pd.DataFrame
        Dataframe of event data including ad/channel details
    event_name : str
        Name of the ad or channel event
    groupby : str | list[str]
        Column(s) to group data by
    title : str
        Plot title

    Returns
    -------
    None
    """

    (
        events_df.groupby(groupby)
                 .size()
                 .div(len(events_df))
                 .plot.bar(
                     ylabel=f"Percent of {event_name}",
                     ylim=(0, 1),
                     color=COLORS,
                     title=title,
                     rot=0,
                 )
    )


def ads_channels_overall_monthly(
    events_df: pd.DataFrame,
    groupby: str,
    title: str,
    ylabel: str,
) -> None:
    """
    Plot monthly percentage distribution of ads/channels for overall
    trends.

    Parameters
    ----------
    events_df : pd.DataFrame
        Dataframe containing event and time data
    groupby : str
        Column to group data by
    title : str
        Plot title
    ylabel : str
        Y-axis label

    Returns
    -------
    None
    """

    months = events_df.time.dt.month
    ax = (
        events_df.groupby([groupby, months])
                 .size()
                 .unstack()
                 .apply(lambda s: s / s.sum())
                 .T.plot.area(xlim=(1, 12), ylim=(0, 1))
    )
    ax.set(
        xticks=list(MONTHS.keys()),
        xticklabels=list(MONTHS.values()),
        xlabel="Month",
        ylabel=ylabel,
        title=title,
    )


def ads_channels_subgroups_annually(
    events_df: pd.DataFrame,
    event_name: str,
    groupby: list[str],
    title: str,
) -> None:
    """
    Plot annual distribution of ads or channels by subgroup.

    Parameters
    ----------
    events_df : pd.DataFrame
        Dataframe containing event and subgroup data
    event_name : str
        Name of the ad/channel event
    groupby : list[str]
        Columns to group data by
    title : str
        Plot title

    Returns
    -------
    None
    """

    (
        events_df.groupby(groupby)
                 .size()
                 .unstack()
                 .apply(lambda s: s / s.sum(), axis=1)
                 .plot(
                     kind="bar",
                     ylabel=f"Percent of {event_name}",
                     ylim=(0, 1),
                     title=title,
                     stacked=True,
                     rot=0,
                 )
    )


def ads_channels_subgroups_monthly(
    subgroups: DataFrameGroupBy,
    groupby: str,
    title: str,
) -> None:
    """
    Plot monthly percentage distribution of ads or channels by subgroup.

    Parameters
    ----------
    subgroups : DataFrameGroupBy
        Grouped dataframe of events by subgroup
    groupby : str
        Column used for grouping
    title : str
        Plot title

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(
        subgroups.ngroups, sharex=True, sharey=True
    )
    for ax, subgroup in zip(axes, subgroups):
        subgroup_name, subgroup_df = subgroup
        months = subgroup_df.time.dt.month
        (
            subgroup_df.groupby([groupby, months])
                       .size()
                       .unstack()
                       .apply(lambda s: s / s.sum())
                       .T.plot.area(ax=ax, xlim=(1, 12), ylim=(0, 1))
        )
        ax.set(
            xticks=list(MONTHS.keys()),
            xticklabels=list(MONTHS.values()),
            xlabel="Month",
            ylabel="%",
            title=f"{subgroups.keys}: {subgroup_name}".replace("_", " "),
        )
    fig.suptitle(title)
    fig.subplots_adjust(hspace=1/2)


def demographic_features_subgroups(
    demographic_features_df: pd.DataFrame,
    subgroups: DataFrameGroupBy,
) -> None:
    """
    Plot the distribution of demographic features by subgroup.

    Parameters
    ----------
    demographic_features_df : pd.DataFrame
        Dataframe of demographic feature values
    subgroups : DataFrameGroupBy
        Grouped dataframe by demographic subgroup

    Returns
    -------
    None
    """

    xmin, xmax = demographic_features_df.stack().agg(["min", "max"])
    xmin = int(np.floor(xmin))
    xmax = int(np.ceil(xmax))
    nbins = (xmax - xmin) * 4 + 1
    bins = np.linspace(xmin, xmax, nbins)
    fig, axes = plt.subplots(
        subgroups.ngroups, sharex=True, sharey=True
   )
    for ax, subgroup in zip(axes, subgroups):
        subgroup_name, subgroup_df = subgroup
        demographic_features = subgroup_df[
            demographic_features_df.columns
        ]
        demographic_features.plot.hist(
            bins=bins,
            xticks=bins,
            xlim=(bins.min(), bins.max()),
            xlabel="Feature values",
            ylabel="",
            label=subgroup_name,
            title=f"{subgroups.keys}: {subgroup_name}".replace("_", " "),
            alpha=1/3,
            rot=90,
            ax=ax,
        )
        feature_means = demographic_features.mean()
        for i, (feature_name, feature_value) in enumerate(
            feature_means.items()
        ):
            ax.axvline(feature_value, color=COLORS[i], ls="dashed")
        ax.legend()
    if subgroups.keys == "income_bin":
        groupby = "income bin"
    elif subgroups.keys == "has_dog":
        groupby = "dog ownership"
    fig.suptitle(f"Distribution of feature values by {groupby}")
    fig.supylabel("Number of people")
    fig.subplots_adjust(hspace=1/2)


def feature_covariances(train_df: pd.DataFrame) -> None:
    """
    Create a scatter matrix to plot covariances between features.

    Parameters
    ----------
    train_df : pd.DataFrame
        Dataframe of training features

    Returns
    -------
    None
    """

    for ax in pd.plotting.scatter_matrix(
        train_df, hist_kwds={"bins": 75}
    ).flat:
        ax.tick_params(axis="both", which="major", labelsize=12)
        fig = ax.get_figure()
        fig.align_xlabels()
        fig.align_ylabels()
        fig.subplots_adjust(hspace=0.05, wspace=0.05)


def best_model_test_set_predictions(
    y_test_exp: np.ndarray[np.float32],
    best_model_test_set_predictions: BEST_MODEL_TEST_SET_PREDICTIONS,
) -> None:
    """
    Plot model predictions against ground truth and shows prediction
    distribution.

    Parameters
    ----------
    y_test_exp : np.ndarray
        Exp-transformed true test values
    best_model_test_set_predictions : BEST_MODEL_TEST_SET_PREDICTIONS
        Dictionary with model names and prediction details

    Returns
    -------
    None
    """

    fig, axes = plt.subplots(
        nrows=len(best_model_test_set_predictions),
        ncols=2,
        sharex=True,
    )
    for model_axes, model in zip(
        axes, best_model_test_set_predictions.items()
    ):
        ax_scatter, ax_hist = model_axes
        model_name, (y_pred_exp, mae) = model
        bins = np.linspace(
            np.floor(min(y_test_exp.min(), y_pred_exp.min())),
            np.ceil(max(y_test_exp.max(), y_pred_exp.max())),
            50,
        )
        ax_hist.hist(y_test_exp, label="y_true", bins=bins, alpha=1/2)
        ax_hist.hist(y_pred_exp, label="y_pred", bins=bins, alpha=1/2)
        ax_hist.legend(loc="center right", frameon=False)
        ax_scatter.scatter(x=y_test_exp, y=y_pred_exp, alpha=1/2, s=5)
        for ax in [ax_scatter, ax_hist]:
            ax.set(
                xlim=0,
                ylim=0,
                title=f"{model_name} (Log MAE = {mae:.4f})",
            )
    fig.subplots_adjust(hspace=2/3)
    fig.supxlabel("Ground truth")
    fig.supylabel("Prediction")
    fig.suptitle("Distribution of predictions by model type")


def linear_model_weights(
    linear_regression_model_weights: np.ndarray[np.float32],
    lasso_model_weights: np.ndarray[np.float32],
    feature_names: pd.Index,
) -> None:
    """
    Plot the feature weight magnitudes for linear and Lasso models.

    Parameters
    ----------
    linear_regression_model_weights : np.ndarray
        Weights from LinearRegression model
    lasso_model_weights : np.ndarray
        Weights from Lasso model
    feature_names : pd.Index
        Names of features used in models

    Returns
    -------
    None
    """

    xticks = np.arange(linear_regression_model_weights.size)
    bar_width = 0.5
    plt.bar(
        xticks - (bar_width / 2),
        linear_regression_model_weights,
        bar_width,
        label="LinearRegression",
    )
    plt.bar(
        xticks + (bar_width / 2),
        lasso_model_weights,
        bar_width,
        label="LassoRegression",
    )
    plt.xticks(xticks, labels=feature_names)
    plt.xlabel("Feature")
    plt.ylabel("Weight magnitude")
    plt.legend()
    plt.title("Linear model feature weights")
    plt.grid(axis="y")


def feature_label_correlations(correlations: pd.Series) -> None:
    """
    Plot the Pearson correlation between features and target.

    Parameters
    ----------
    correlations : pd.Series
        Series of feature-to-target correlations

    Returns
    -------
    None
    """

    correlations.plot.bar(
        xlabel="Feature",
        ylabel="Pearson correlation",
        ylim=(0, 1),
        rot=0,
    ).grid(axis="y")
