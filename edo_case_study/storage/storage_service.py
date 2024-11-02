"""
Defines the generic StorageService interface for handling various
storage backends.

This module provides an abstract base class, `StorageService`, which
defines general-purpose methods for saving, loading, deleting, listing,
and checking the existence of data. Specific storage backends (e.g.,
local filesystem, S3, SQL database) should implement this interface,
providing their own backend-specific logic.
"""

from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
from typing import Any, Callable


class StorageService(ABC):
    """
    Generic interface for a storage service supporting save, load,
    delete, list, and existence-check operations.

    Intended to supports various backends such as local filesystem, S3,
    and SQL databases.
    """

    _supported_data_types = set()

    @abstractmethod
    def save(self, data: Any, path: Path | str, **kwargs) -> None:
        """
        Save data to the specified path

        Parameters
        ----------
        data : Any
            The data to save
        path : Path | str
            The path to save the data

        Returns
        -------
        None
        """

    @abstractmethod
    def load(self, path: Path | str, **kwargs) -> Any:
        """
        Load data from the specified path

        Parameters
        ----------
        path : Path | str
            The path to load the data from

        Returns
        -------
        Any
            The loaded data
        """

    @abstractmethod
    def delete(self, path: Path | str) -> None:
        """
        Delete the specified path

        Parameters
        ----------
        path : Path | str
            The path to delete

        Returns
        -------
        None
        """

    @abstractmethod
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

    @abstractmethod
    def list(self, directory: Path | str) -> list[str]:
        """
        List all items in the specified directory

        Parameters
        ----------
        directory : Path | str
            The directory to list items from

        Returns
        -------
        list[str]
            List of items in the directory
        """

    @property
    def supported_data_types(self) -> set[str]:
        return self._supported_data_types

    @staticmethod
    def validate_data_type(func: Callable) -> Callable:
        """
        Decorator to validate that the specified data type is supported
        by the storage service before executing the decorated method.

        Parameters
        ----------
        func : Callable
            The method to be decorated

        Returns
        -------
        Callable
            The wrapped method with data type validation
        """
        @wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            if "data_type" in kwargs:
                data_type = kwargs["data_type"]
                if data_type not in self.supported_data_types:
                    raise ValueError(
                        f"Data type {data_type!r} not recognized. "
                        f"Only {self.supported_data_types} are "
                        "supported."
                    )
            return func(self, *args, **kwargs)

        return wrapper
