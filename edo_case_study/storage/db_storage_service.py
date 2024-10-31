"""
Implements the DBStorageService for saving/loading data in a SQL
database.

This module provides a concrete implementation of the `StorageService`
interface, designed for SQL database storage using SQLAlchemy. It
supports saving/loading both pandas DataFrames and arbitrary Python
objects (as binary blobs) in a SQL table. Private methods handle
backend-specific behavior, allowing the public interface to remain
consistent across different storage backends.
"""

import pickle
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect
from sqlalchemy.engine.base import Engine

from edo_case_study.storage.storage_service import StorageService


SUPPORTED_DATA_TYPES: set[str] = {
    "dataframe",
    "object",
}


class DBStorageService(StorageService):
    """
    SQL database storage service implementation for saving/loading data
    in a SQL database.
    """

    _supported_data_types = SUPPORTED_DATA_TYPES

    def __init__(self, connection_string: str = "sqlite:///:memory:"):
        """
        Initialize with a database connection string

        Parameters
        ----------
        connection_string : str, optional
            Database connection string in SQLAlchemy format

        Returns
        -------
        None
        """

        self.engine: Engine = create_engine(connection_string)

    @StorageService.validate_data_type
    def save(
        self,
        data: Any,
        table_name: str,
        data_type: str = "object",
        **kwargs,
    ) -> None:
        """
        Save data to a SQL table in a format based on the data type

        Parameters
        ----------
        data : Any
            The data to save
        table_name : str
            The table name to save the data to
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
            self._save_dataframe(data, table_name, **kwargs)
        else:
            self._save_object(data, table_name, **kwargs)

    @StorageService.validate_data_type
    def load(
        self,
        table_name: str,
        data_type: str = "object",
        **kwargs,
    ) -> Any:
        """
        Load data from a SQL table based on the data type

        Parameters
        ----------
        table_name : str
            The table name to load data from
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
            return self._load_dataframe(table_name)
        else:
            return self._load_object(table_name, **kwargs)

    def delete(
        self,
        table_name: str,
    ) -> None:
        """
        Delete the specified table from the database

        Parameters
        ----------
        table_name : str
            The table name to delete

        Returns
        -------
        None
        """

        with self.engine.connect() as connection:
            connection.execute(f"DROP TABLE IF EXISTS {table_name}")

    def exists(self, table_name: str) -> bool:
        """
        Check if the specified table exists in the database

        Parameters
        ----------
        table_name : str
            The table name to check for existence

        Returns
        -------
        bool
            True if the table exists, False otherwise
        """

        inspector = inspect(self.engine)
        return inspector.has_table(table_name)

    def list(self, directory: str = "") -> list[str]:
        """
        List all tables in the database

        Parameters
        ----------
        directory : str, optional
            Placeholder for compatibility with other storage services,
            not used in database context

        Returns
        -------
        list of str
            List of table names in the database
        """

        inspector = inspect(self.engine)
        return inspector.get_table_names()

    def _save_dataframe(
        self,
        data: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> None:
        """
        Save a pandas DataFrame to a SQL table

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to save
        table_name : str
            The table name to save the DataFrame to
        if_exists : str, optional
            The action if table exists: 'fail', 'replace', 'append', by
            default 'replace'

        Returns
        -------
        None
        """

        data.to_sql(
            table_name, self.engine, if_exists=if_exists, index=False
        )

    def _load_dataframe(self, table_name: str) -> pd.DataFrame:
        """
        Load a pandas DataFrame from a SQL table

        Parameters
        ----------
        table_name : str
            The table name to load the DataFrame from

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame
        """

        return pd.read_sql(f"SELECT * FROM {table_name}", self.engine)

    def _save_object(
        self,
        obj: Any,
        table_name: str,
        key_column: str = "id",
        key_value: int = 1,
    ) -> None:
        """
        Save an arbitrary Python object to a SQL table as a binary blob

        Parameters
        ----------
        obj : Any
            The object to save
        table_name : str
            The table name to save the object to
        key_column : str, optional
            The primary key column for identification, by default 'id'
        key_value : int, optional
            The primary key value, by default 1

        Returns
        -------
        None
        """

        with self.engine.connect() as connection:
            serialized_obj = pickle.dumps(obj)
            connection.execute(
                f"CREATE TABLE IF NOT EXISTS {table_name} ({key_column} "
                "INT PRIMARY KEY, data BLOB)"
            )
            connection.execute(
                f"REPLACE INTO {table_name} ({key_column}, data) VALUES "
                "(:key_value, :data)",
                {"key_value": key_value, "data": serialized_obj},
            )

    def _load_object(
        self,
        table_name: str,
        key_column: str = "id",
        key_value: int = 1,
    ) -> Any:
        """
        Load an arbitrary Python object from a SQL table as a binary
        blob

        Parameters
        ----------
        table_name : str
            The table name to load the object from
        key_column : str, optional
            The primary key column for identification, by default 'id'
        key_value : int, optional
            The primary key value, by default 1

        Returns
        -------
        Any
            The loaded object

        Raises
        ------
        ValueError
            If no object is found with the specified key
        """

        query = (
            f"SELECT data FROM {table_name} WHERE {key_column} = "
            ":key_value"
        )
        with self.engine.connect() as connection:
            result = connection.execute(
                query, {"key_value": key_value}
            ).fetchone()
            if result:
                return pickle.loads(result[0])
            else:
                raise ValueError(
                    f"No object found with {key_column}={key_value}"
                )
