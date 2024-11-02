"""
Implements the S3StorageService for saving/loading data in an S3
bucket.

This module provides a concrete implementation of the `StorageService`
interface, designed for Amazon S3 storage. It supports saving/loading
both pandas DataFrames and arbitrary Python objects in S3, and uses
boto3 for S3 operations. Private methods handle backend-specific
behavior, allowing the public interface to remain consistent across
different storage backends.
"""

from io import BytesIO
from pathlib import Path
import pickle
from typing import Any

import boto3
import pandas as pd

from edo_case_study.storage.storage_service import StorageService


SUPPORTED_DATA_TYPES: set[str] = {
    "dataframe",
    "object",
}


class S3StorageService(StorageService):
    """
    S3 storage service implementation for saving/loading data in an S3
    bucket
    """

    _supported_data_types = SUPPORTED_DATA_TYPES

    def __init__(self, bucket_name: str, s3_client=None):
        """
        Initialize with S3 bucket name and optional S3 client

        Parameters
        ----------
        bucket_name : str
            The S3 bucket name
        s3_client : optional
            The S3 client object

        Returns
        -------
        None
        """

        self.bucket_name = bucket_name
        self.s3_client = s3_client or boto3.client("s3")

    @StorageService.validate_data_type
    def save(
        self,
        data: Any,
        path: Path | str,
        data_type: str = "object",
    ) -> None:
        """
        Save data to an S3 object in a format based on the data type

        Parameters
        ----------
        data : Any
            The data to save
        path : Path | str
            The S3 object path to save the data
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
        Load data from an S3 object based on the data type

        Parameters
        ----------
        path : Path | str
            The S3 object path from which to load the data
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
        Delete the specified file from the S3 bucket

        Parameters
        ----------
        path : Path | str
            The S3 object path to delete

        Returns
        -------
        None
        """

        self.s3_client.delete_object(Bucket=self.bucket_name, Key=path)

    def exists(self, path: Path | str) -> bool:
        """
        Check if the specified file exists in the S3 bucket

        Parameters
        ----------
        path : Path | str
            The S3 object path to check for existence

        Returns
        -------
        bool
            True if the file exists, False otherwise
        """

        try:
            self.s3_client.head_object(
                Bucket=self.bucket_name, Key=path
            )
            return True
        except self.s3_client.exceptions.ClientError:
            return False
        except self.s3_client.exceptions.NoSuchKey:
            return False

    def list(self, directory: Path | str) -> list[str]:
        """
        List all files in the specified directory within the S3 bucket

        Parameters
        ----------
        directory : Path | str
            The S3 directory to list files from

        Returns
        -------
        list[str]
            List of file paths in the directory
        """

        file_keys = []
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(
            Bucket=self.bucket_name, Prefix=directory
        ):
            if "Contents" in page:
                for obj in page["Contents"]:
                    file_keys.append(obj["Key"])
        return file_keys

    def _save_dataframe(
        self,
        data: pd.DataFrame,
        path: Path | str,
    ) -> None:
        """
        Save a pandas DataFrame to an S3 object as a CSV file

        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame to save
        path : Path | str
            The S3 object path to save the DataFrame

        Returns
        -------
        None
        """

        csv_buffer = BytesIO()
        data.to_csv(csv_buffer, index=False)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=path,
            Body=csv_buffer.getvalue(),
        )

    def _load_dataframe(self, path: Path | str) -> pd.DataFrame:
        """
        Load a pandas DataFrame from an S3 object as a CSV file

        Parameters
        ----------
        path : Path | str
            The S3 object path from which to load the DataFrame

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame
        """

        response = self.s3_client.get_object(
            Bucket=self.bucket_name, Key=path
        )
        return pd.read_csv(response["Body"])

    def _save_object(self, obj: Any, path: Path | str) -> None:
        """
        Save an arbitrary Python object to an S3 object using pickle

        Parameters
        ----------
        obj : Any
            The object to save
        path : Path | str
            The S3 object path to save the object

        Returns
        -------
        None
        """

        obj_buffer = BytesIO()
        pickle.dump(obj, obj_buffer)
        self.s3_client.put_object(
            Bucket=self.bucket_name,
            Key=path,
            Body=obj_buffer.getvalue(),
        )

    def _load_object(self, path: Path | str) -> Any:
        """
        Load an arbitrary Python object from an S3 object using pickle

        Parameters
        ----------
        path : Path | str
            The S3 object path from which to load the object

        Returns
        -------
        Any
            The loaded object
        """

        response = self.s3_client.get_object(
            Bucket=self.bucket_name, Key=path
        )
        return pickle.load(BytesIO(response["Body"].read()))
