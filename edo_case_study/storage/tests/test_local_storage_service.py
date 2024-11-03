"""
Unit tests for the LocalStorageService class.
"""

from pathlib import Path
import pickle

import pandas as pd
import pytest

from edo_case_study.storage.local_storage_service import LocalStorageService


@pytest.fixture
def storage_service():
    """Fixture to provide LocalStorageService instance."""
    return LocalStorageService()


def test_save_dataframe(storage_service, tmp_path):
    """Test saving a DataFrame to a CSV file."""
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    path = tmp_path / "test.csv"
    storage_service.save(data, path, data_type="dataframe")
    assert path.exists()
    loaded_data = pd.read_csv(path)
    pd.testing.assert_frame_equal(data, loaded_data)


def test_save_object(storage_service, tmp_path):
    """Test saving a Python object with pickle."""
    data = {"key": "value"}
    path = tmp_path / "test.pkl"
    storage_service.save(data, path, data_type="object")
    assert path.exists()
    with path.open("rb") as f:
        loaded_data = pickle.load(f)
    assert data == loaded_data


def test_load_dataframe(storage_service, tmp_path):
    """Test loading a DataFrame from a CSV file."""
    path = tmp_path / "test.csv"
    data = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    data.to_csv(path, index=False)
    loaded_data = storage_service.load(path, data_type="dataframe")
    pd.testing.assert_frame_equal(data, loaded_data)


def test_load_object(storage_service, tmp_path):
    """Test loading a Python object with pickle."""
    path = tmp_path / "test.pkl"
    data = {"key": "value"}
    with path.open("wb") as f:
        pickle.dump(data, f)
    loaded_data = storage_service.load(path, data_type="object")
    assert data == loaded_data


def test_delete(storage_service, tmp_path):
    """Test deletion of a specified file."""
    path = tmp_path / "test.txt"
    path.touch()
    assert path.exists()
    storage_service.delete(path)
    assert not path.exists()


def test_exists(storage_service, tmp_path):
    """Test existence check of a specified path."""
    path = tmp_path / "test.txt"
    path.touch()
    assert storage_service.exists(path)
    path.unlink()
    assert not storage_service.exists(path)


def test_list(storage_service, tmp_path):
    """Test listing files in a directory."""
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    expected_files = {
        str(tmp_path / "file1.txt"), str(tmp_path / "file2.txt")
    }
    listed_files = set(storage_service.list(tmp_path))
    assert listed_files == expected_files


@pytest.mark.parametrize("invalid_data_type", ["invalid", ""], ids=str)
def test_save_invalid_data_type(
    invalid_data_type,
    storage_service,
    tmp_path,
):
    """Test calling a method with an unsupported data type."""
    data = {"key": "value"}
    path = tmp_path / "test.pkl"
    with pytest.raises(
        ValueError,
        match=f"Data type {invalid_data_type!r} not recognized"
    ):
        storage_service.save(data, path, data_type=invalid_data_type)


def test_save_dataframe_invalid_path(storage_service):
    """Test save with invalid path for a DataFrame."""
    data = pd.DataFrame({"col1": [1, 2]})
    invalid_path = "/invalid_path/test.csv"
    with pytest.raises(OSError):
        storage_service.save(data, invalid_path, data_type="dataframe")


def test_save_object_invalid_path(storage_service):
    """Test save with invalid path for a Python object."""
    data = {"key": "value"}
    invalid_path = "/invalid_path/test.pkl"
    with pytest.raises(OSError):
        storage_service.save(data, invalid_path, data_type="object")


def test_load_nonexistent_file(storage_service, tmp_path):
    """Test load of a nonexistent file."""
    non_existent_path = tmp_path / "nonexistent.pkl"
    with pytest.raises(FileNotFoundError):
        storage_service.load(non_existent_path, data_type="object")


def test_load_empty_file_as_dataframe(storage_service, tmp_path):
    """Test load of an empty file as a DataFrame."""
    empty_path = tmp_path / "empty.csv"
    empty_path.touch()
    with pytest.raises(pd.errors.EmptyDataError):
        storage_service.load(empty_path, data_type="dataframe")


def test_load_empty_file_as_object(storage_service, tmp_path):
    """Test load of an empty file as a Python object."""
    empty_path = tmp_path / "empty.pkl"
    empty_path.touch()
    with pytest.raises(EOFError):
        storage_service.load(empty_path, data_type="object")


@pytest.mark.parametrize("invalid_data_type", ["invalid", ""], ids=repr)
def test_load_invalid_data_type(
    invalid_data_type,
    storage_service,
    tmp_path,
):
    """Test calling a method with an unsupported data type."""
    with pytest.raises(
        ValueError,
        match=f"Data type {invalid_data_type!r} not recognized"
    ):
        storage_service.load(tmp_path, data_type=invalid_data_type)


def test_delete_nonexistent_file(storage_service, tmp_path):
    """Test deletion of a nonexistent file."""
    nonexistent_path = tmp_path / "nonexistent.txt"
    storage_service.delete(nonexistent_path)
    assert not nonexistent_path.exists()


def test_exists_nonexistent_file(storage_service, tmp_path):
    """Test existence check for a nonexistent file."""
    nonexistent_path = tmp_path / "nonexistent.txt"
    assert not storage_service.exists(nonexistent_path)


def test_list_empty_directory(storage_service, tmp_path):
    """Test listing files in an empty directory."""
    assert storage_service.list(tmp_path) == []


def test_list_directory_with_subdirectories(storage_service, tmp_path):
    """Test listing only files in the specified directory."""
    (tmp_path / "file1.txt").touch()
    (tmp_path / "file2.txt").touch()
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "file3.txt").touch()
    listed_files = set(storage_service.list(tmp_path))
    expected_files = {
        str(tmp_path / "file1.txt"), str(tmp_path / "file2.txt")
    }
    assert listed_files == expected_files


def test_save_load_large_dataframe(storage_service, tmp_path):
    """Test saving and loading a large DataFrame."""
    large_data = pd.DataFrame(
        {"col1": range(10000), "col2": range(10000)}
    )
    path = tmp_path / "large_test.csv"
    storage_service.save(large_data, path, data_type="dataframe")
    assert path.exists()
    loaded_data = storage_service.load(path, data_type="dataframe")
    pd.testing.assert_frame_equal(large_data, loaded_data)


def test_save_load_large_object(storage_service, tmp_path):
    """Test saving and loading a large Python object."""
    large_data = {f"key_{i}": i for i in range(10000)}
    path = tmp_path / "large_test.pkl"
    storage_service.save(large_data, path, data_type="object")
    assert path.exists()
    loaded_data = storage_service.load(path, data_type="object")
    assert large_data == loaded_data
