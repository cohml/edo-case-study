"""
Implements the LocalStorageService for saving/loading data on the local
filesystem.

This module provides a concrete implementation of the `StorageService`
interface, designed for local filesystem storage. It supports
saving/loading both pandas DataFrames and arbitrary Python objects,
with methods that use pathlib for path management. Private methods
handle backend-specific behavior, allowing the public interface to
remain consistent across different storage backends.
"""

from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from edo_case_study.storage.storage_service import StorageService


SUPPORTED_DATA_TYPES: set[str] = {
    "dataframe",
    "object",
}


class LocalStorageService(StorageService):
    """
    Local storage service implementation for saving/loading data on the
    filesystem.
    """

    _supported_data_types = SUPPORTED_DATA_TYPES

    @StorageService.validate_data_type
    def save(
        self,
        data: Any,
        path: Path | str,
        data_type: str = "object",
    ) -> None:
        """
        Save data to a local file in a format based on the data type

        Parameters
        ----------
        data : Any
            The data to save
        path : Path | str
            The path to save the data
        data_type : str, optional
            The type of data to save ('dataframe' or 'object'), by
            default 'object'

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the specified data type is not recognized
        """

        if data_type == "dataframe":
            self._save_dataframe(data, path)
        else:
            self._save_object(data, path)

    @StorageService.validate_data_type
    def load(self, path: Path | str, data_type: str = "object") -> Any:
        """
        Load data from a local file based on the data type

        Parameters
        ----------
        path : Path | str
            The path to load the data from
        data_type : str, optional
            The type of data to load ('dataframe' or 'object'), by
            default 'object'

        Returns
        -------
        Any
            The loaded data

        Raises
        ------
        ValueError
            If the specified data type is not recognized
        """

        if data_type == "dataframe":
            return self._load_dataframe(path)
        else:
            return self._load_object(path)

    def delete(self, path: Path | str) -> None:
        """
        Delete the specified file

        Parameters
        ----------
        path : Path | str
            The path of the file to delete

        Returns
        -------
        None
        """

        Path(path).unlink(missing_ok=True)

    def exists(self, path: Path | str) -> bool:
        """
        Check if the specified path exists

        Parameters
        ----------
        path : Path | str
            The path to check for existence

        Returns
        -------
        bool
            True if the path exists, False otherwise
        """

        return Path(path).exists()

    def list(self, directory: Path | str) -> list[str]:
        """
        List all files in the specified directory

        Parameters
        ----------
        directory : Path | str
            The directory to list files from

        Returns
        -------
        list[str]
            List of file paths in the directory
        """

        return [
            str(file) for file in Path(directory).iterdir()
            if file.is_file()
        ]

    def _save_dataframe(
        self,
        data: pd.DataFrame,
        path: Path | str,
    ) -> None:
        """
        Save a pandas DataFrame to a local file in CSV format

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to save
        path : Path | str
            The path to save the DataFrame

        Returns
        -------
        None
        """

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)

    def _load_dataframe(self, path: Path | str) -> pd.DataFrame:
        """
        Load a pandas DataFrame from a local CSV file

        Parameters
        ----------
        path : Path | str
            The path from which to load the DataFrame

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame
        """

        return pd.read_csv(path)

    def _save_object(self, obj: Any, path: Path | str) -> None:
        """
        Save an arbitrary Python object to a local file using pickle

        Parameters
        ----------
        obj : Any
            The object to save
        path : Path | str
            The path to save the object

        Returns
        -------
        None
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(obj, f)

    def _load_object(self, path: Path | str) -> Any:
        """
        Load an arbitrary Python object from a local file using pickle

        Parameters
        ----------
        path : Path | str
            The path from which to load the object

        Returns
        -------
        Any
            The loaded object
        """

        with Path(path).open("rb") as f:
            return pickle.load(f)
