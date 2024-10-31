"""
This module contains utility functions that support data loading,
preprocessing, and display, as well as statistical checks for outliers.
"""

from calendar import month_abbr
from pathlib import Path

from IPython.display import HTML, display
import numpy as np
import pandas as pd
from scipy.stats import zscore


BEST_MODEL_TEST_SET_PREDICTIONS = tuple[
    np.ndarray[np.float32],
    dict[str, tuple[np.ndarray[np.float32], np.float32]]
]
COLORS = ["tab:blue", "tab:orange", "tab:green"]
MONTHS = {i: month_abbr[i] for i in range(1, 13)}


def count_events(
    df: pd.DataFrame,
    event_name: str,
    show=False,
) -> pd.DataFrame:
    """
    Count the number of events per person.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing event data
    event_name : str
        Name of the event being counted
    show : bool, optional
        If True, displays the count dataframe, by default False

    Returns
    -------
    pd.DataFrame
        Dataframe of event counts per person
    """

    counts_df = df.groupby("person").size()
    if show:
        display(counts_df.head().to_frame(f"n_{event_name}"))
    return counts_df


def display_dfs_side_by_side(dfs: dict[str, pd.DataFrame]) -> None:
    """
    Display dataframes side by side in an output cell for comparison.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary of dataframes to display with captions

    Returns
    -------
    None
    """

    whitespace_amount_between_dfs = 30 // len(dfs)
    display_str = ""
    for header, df in dfs.items():
        display_str += (
            df.head()
              .style
              .set_table_attributes("style='display:inline'")
              .set_caption(f"{header}: {len(df):,} rows".upper())
              ._repr_html_()
        )
        display_str += "\xa0" * whitespace_amount_between_dfs
    display(HTML(display_str))


def is_outlier(s: pd.Series) -> pd.Series:
    """
    Identify outliers using the z-score.

    Parameters
    ----------
    s : pd.Series
        Series of data to check for outliers

    Returns
    -------
    pd.Series
        Boolean series indicating outlier rows
    """

    z = abs(zscore(s))
    return z > 3


def join_dfs(
    dfs: dict[str, pd.DataFrame],
    right_key: str,
) -> pd.DataFrame:
    """
    Join two dataframes on ``person`` and transform time columns.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Dictionary of dataframes to join
    right_key : str
        Key in dictionary to join with the ``people`` dataframe

    Returns
    -------
    pd.DataFrame
        Joined dataframe with time parsed and added month column
    """

    joined_df = dfs["people"].merge(dfs[right_key], on="person")
    if "time" in joined_df.columns:
        joined_df.time = pd.to_datetime(joined_df.time)
        joined_df["month"] = joined_df.time.dt.month.map(MONTHS)
    display(joined_df.head())
    return joined_df


def load_dfs(data_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all CSV files from the specified directory into dataframes.

    Parameters
    ----------
    data_dir : str
        Directory containing CSV files

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary with file basenames as keys and dataframes as values
    """

    dfs = {
        csv.stem: pd.read_csv(csv) for csv in Path(data_dir).glob("*.csv")
    }
    display_dfs_side_by_side(dfs)
    return dfs
