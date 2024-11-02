# EDO Case Study: Data Storage Services

This set of modules provides a unified interface for handling data storage
across multiple backends, including SQL databases, the local filesystem, and
Amazon S3. Designed to integrate with a modeling pipeline, the storage services
support saving, loading, deleting, listing, and existence-checking for various
data types, specifically pandas DataFrames and arbitrary Python objects. Each
backend offers its own implementation of the `StorageService` interface, making
it simple to switch between storage options as needed.

## Project Overview

### Modules

- **`DBStorageService`**: Manages storage in a SQL database using SQLAlchemy.
  Supports saving/loading of both DataFrames and Python objects (stored as
  binary blobs). The default connection string points to an in-memory SQLite
  database, but any SQLAlchemy-compatible database can be used.

- **`LocalStorageService`**: Manages storage on the local filesystem using the
  `pathlib` library. DataFrames are saved as CSV files, and Python objects are
  stored as serialized `pickle` files.

- **`S3StorageService`**: Manages storage in an Amazon S3 bucket using `boto3`.
  DataFrames are saved as CSV files in S3, and arbitrary Python objects are
  stored as serialized blobs. The S3 client is configurable, allowing for
  integration with various AWS configurations.

- **`StorageService`**: An abstract base class defining the interface for
  storage services. It enforces consistency across backends and provides
  essential methods (i.e., `save`, `load`, `delete`, `exists`, `list`) to
  manage data.

### Key Features

1. **Unified Interface**: Each backend implements the same set of methods, so
   switching between storage types is seamless. This is particularly helpful in
   pipelines that may need to switch between local and cloud storage depending
   on the deployment environment.

2. **Data Type Validation**: Each storage service validates the specified data
   type to ensure compatibility. Currently supported data types include
   "dataframe" and "object", but this can be extended as new data types become
   necessary.

3. **Backend-Specific Behavior**: Each backend has methods optimized for its
   storage type. For instance, the database service can load and save data as
   binary blobs in addition to CSVs, while the S3 service manages data via
   Amazon's S3 storage protocol.

4. **Modular Design**: Additional storage backends can be added by implementing
   the `StorageService` interface, making the storage module adaptable to
   different infrastructure setups.

## Installation

To set up the environment required to run these storage services, use the
provided `environment.yaml` file. From the project’s root directory, execute
the following command to create the conda environment:

```shell
conda env create --file environment.yaml
```

Once the environment is set up, activate it using:

```shell
conda activate edo-case-study
```

## Tests

As per the instructions for this assignment, a suite of unit tests has been
written for the `LocalStorageService` class. After activating the environment,
these tests can be run using:

```shell
pytest tests/test_local_storage_service.py
```
