"""
Tests for ingenious.utils.load_sample_data module
"""

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

from ingenious.utils.load_sample_data import sqlite_sample_db


class TestSqliteSampleDb:
    """Test cases for sqlite_sample_db class"""

    @patch("ingenious.utils.load_sample_data.get_config")
    @patch("ingenious.utils.load_sample_data.os.path.exists")
    @patch("ingenious.utils.load_sample_data.os.makedirs")
    @patch("ingenious.utils.load_sample_data.sqlite3.connect")
    @patch("ingenious.utils.load_sample_data.pd.read_csv")
    def test_init_with_csv_file_exists(
        self, mock_read_csv, mock_connect, mock_makedirs, mock_exists, mock_get_config
    ):
        """Test initialization when CSV file exists"""
        # Setup mocks
        mock_config = Mock()
        mock_config.local_sql_db.database_path = "/tmp/test.db"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_csv_path = "/tmp/test.csv"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_database_name = "test_table"
        mock_get_config.return_value = mock_config

        mock_exists.side_effect = lambda path: path in ["/tmp", "/tmp/test.csv"]  # nosec B108: acceptable for testing

        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None
        mock_connect.return_value = mock_connection

        # Mock DataFrame with to_sql method
        mock_df = Mock(spec=pd.DataFrame)
        mock_df.to_sql = Mock()
        mock_df.dtypes = pd.Series(
            {"name": "object", "age": "int64", "score": "float64"}
        )
        mock_read_csv.return_value = mock_df

        # Initialize
        db = sqlite_sample_db()

        # Verify initialization
        assert db.db_path == "/tmp/test.db"  # nosec B108: acceptable for testing
        assert db._config == mock_config
        mock_makedirs.assert_not_called()  # Directory exists
        mock_connect.assert_called_with("/tmp/test.db", check_same_thread=False)  # nosec B108: acceptable for testing

    @patch("ingenious.utils.load_sample_data.get_config")
    @patch("ingenious.utils.load_sample_data.os.path.exists")
    @patch("ingenious.utils.load_sample_data.os.makedirs")
    @patch("ingenious.utils.load_sample_data.sqlite3.connect")
    @patch("ingenious.utils.load_sample_data.pd.read_csv")
    def test_init_with_directory_creation(
        self, mock_read_csv, mock_connect, mock_makedirs, mock_exists, mock_get_config
    ):
        """Test initialization when directory needs to be created"""
        # Setup mocks
        mock_config = Mock()
        mock_config.local_sql_db.database_path = "/tmp/new_dir/test.db"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_csv_path = "/tmp/test.csv"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_database_name = "test_table"
        mock_get_config.return_value = mock_config

        mock_exists.side_effect = (
            lambda path: path == "/tmp/test.csv"  # nosec B108: acceptable for testing
        )  # Only CSV exists

        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None
        mock_connect.return_value = mock_connection

        # Mock DataFrame with to_sql method
        mock_df = Mock(spec=pd.DataFrame)
        mock_df.to_sql = Mock()
        mock_df.dtypes = pd.Series({"name": "object", "age": "int64"})
        mock_read_csv.return_value = mock_df

        # Initialize
        sqlite_sample_db()

        # Verify directory creation
        mock_makedirs.assert_called_once_with("/tmp/new_dir", exist_ok=True)  # nosec B108: acceptable for testing

    @patch("ingenious.utils.load_sample_data.get_config")
    @patch("ingenious.utils.load_sample_data.os.path.exists")
    @patch("ingenious.utils.load_sample_data.sqlite3.connect")
    def test_init_without_csv_file(self, mock_connect, mock_exists, mock_get_config):
        """Test initialization when CSV file doesn't exist (fallback table)"""
        # Setup mocks
        mock_config = Mock()
        mock_config.local_sql_db.database_path = "/tmp/test.db"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_csv_path = "/tmp/nonexistent.csv"  # nosec B108: acceptable for testing
        mock_get_config.return_value = mock_config

        mock_exists.side_effect = (
            lambda path: path == "/tmp"
        )  # Only directory exists  # nosec B108: acceptable for testing

        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None
        mock_connect.return_value = mock_connection

        # Initialize
        sqlite_sample_db()

        # Verify fallback table creation
        mock_connection.execute.assert_called()
        call_args = mock_connection.execute.call_args[0][0]
        assert "students_performance" in call_args
        assert "gender TEXT" in call_args

    def test_execute_sql_with_results(self):
        """Test execute_sql method expecting results"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        try:
            # Setup test data
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
            conn.commit()
            conn.close()

            # Create instance with minimal mocking
            with patch(
                "ingenious.utils.load_sample_data.get_config"
            ) as mock_get_config:
                mock_config = Mock()
                mock_config.local_sql_db.database_path = "/tmp/dummy.db"  # nosec B108: acceptable for testing
                mock_config.local_sql_db.sample_csv_path = "/nonexistent.csv"
                mock_get_config.return_value = mock_config

                with patch(
                    "ingenious.utils.load_sample_data.os.path.exists",
                    return_value=False,
                ):
                    with patch(
                        "ingenious.utils.load_sample_data.sqlite3.connect"
                    ) as mock_connect:
                        mock_connection = MagicMock()
                        mock_connection.__enter__.return_value = mock_connection
                        mock_connection.__exit__.return_value = None
                        mock_connect.return_value = mock_connection
                        db = sqlite_sample_db()

                # Override db_path for the actual test
                db.db_path = db_path
                result = db.execute_sql("SELECT * FROM test ORDER BY id")

                assert result is not None
                assert len(result) == 2
                assert result[0]["id"] == 1
                assert result[0]["name"] == "Alice"
                assert result[1]["id"] == 2
                assert result[1]["name"] == "Bob"

        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_execute_sql_without_results(self):
        """Test execute_sql method not expecting results"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        try:
            # Setup test data
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.commit()
            conn.close()

            # Create instance with minimal mocking
            with patch(
                "ingenious.utils.load_sample_data.get_config"
            ) as mock_get_config:
                mock_config = Mock()
                mock_config.local_sql_db.database_path = "/tmp/dummy.db"  # nosec B108: acceptable for testing
                mock_config.local_sql_db.sample_csv_path = "/nonexistent.csv"
                mock_get_config.return_value = mock_config

                with patch(
                    "ingenious.utils.load_sample_data.os.path.exists",
                    return_value=False,
                ):
                    with patch(
                        "ingenious.utils.load_sample_data.sqlite3.connect"
                    ) as mock_connect:
                        mock_connection = MagicMock()
                        mock_connection.__enter__.return_value = mock_connection
                        mock_connection.__exit__.return_value = None
                        mock_connect.return_value = mock_connection
                        db = sqlite_sample_db()

                # Override db_path for the actual test
                db.db_path = db_path
                result = db.execute_sql(
                    "INSERT INTO test VALUES (3, 'Charlie')", expect_results=False
                )

                assert result is None

                # Verify the insert worked
                conn = sqlite3.connect(db_path)
                cursor = conn.execute("SELECT COUNT(*) FROM test")
                count = cursor.fetchone()[0]
                conn.close()
                assert count == 1

        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)

    def test_execute_sql_with_error(self):
        """Test execute_sql method with database error"""
        with patch("ingenious.utils.load_sample_data.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.local_sql_db.database_path = "/tmp/dummy.db"  # nosec B108: acceptable for testing
            mock_config.local_sql_db.sample_csv_path = "/nonexistent.csv"
            mock_get_config.return_value = mock_config

            with patch(
                "ingenious.utils.load_sample_data.os.path.exists", return_value=False
            ):
                with patch(
                    "ingenious.utils.load_sample_data.sqlite3.connect"
                ) as mock_connect:
                    mock_connection = MagicMock()
                    mock_connection.__enter__.return_value = mock_connection
                    mock_connection.__exit__.return_value = None
                    mock_connect.return_value = mock_connection
                    db = sqlite_sample_db()

            # Mock connection that raises error for execute_sql method
            with patch(
                "ingenious.utils.load_sample_data.sqlite3.connect"
            ) as mock_connect:
                mock_connection = Mock()
                mock_cursor = Mock()
                mock_cursor.execute.side_effect = sqlite3.Error("Test error")
                mock_connection.cursor.return_value = mock_cursor
                mock_connect.return_value = mock_connection

                with patch("builtins.print") as mock_print:
                    result = db.execute_sql("SELECT * FROM test")

                    assert result is None
                    mock_print.assert_called_once()
                    mock_connection.close.assert_called_once()

    @patch("ingenious.utils.load_sample_data.get_config")
    @patch("ingenious.utils.load_sample_data.os.path.exists")
    @patch("ingenious.utils.load_sample_data.sqlite3.connect")
    @patch("ingenious.utils.load_sample_data.pd.read_csv")
    def test_create_table_with_different_dtypes(
        self, mock_read_csv, mock_connect, mock_exists, mock_get_config
    ):
        """Test _create_table with different pandas dtypes"""
        # Setup mocks
        mock_config = Mock()
        mock_config.local_sql_db.database_path = "/tmp/test.db"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_csv_path = "/tmp/test.csv"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_database_name = "test_table"
        mock_get_config.return_value = mock_config

        mock_exists.side_effect = lambda path: path in ["/tmp", "/tmp/test.csv"]  # nosec B108: acceptable for testing

        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None
        mock_connect.return_value = mock_connection

        # Mock DataFrame with different dtypes
        mock_df = Mock(spec=pd.DataFrame)
        mock_df.to_sql = Mock()
        mock_df.dtypes = pd.Series(
            {"int_col": "int64", "float_col": "float64", "str_col": "object"}
        )
        mock_read_csv.return_value = mock_df

        # Initialize
        with patch("builtins.print") as mock_print:
            sqlite_sample_db()

            # Verify table creation call
            create_call = mock_connection.execute.call_args_list[0][0][0]
            assert "int_col INTEGER" in create_call
            assert "float_col REAL" in create_call
            assert "str_col TEXT" in create_call
            mock_print.assert_called()

    @patch("ingenious.utils.load_sample_data.get_config")
    @patch("ingenious.utils.load_sample_data.os.path.exists")
    @patch("ingenious.utils.load_sample_data.sqlite3.connect")
    @patch("ingenious.utils.load_sample_data.pd.read_csv")
    def test_load_csv_data_success(
        self, mock_read_csv, mock_connect, mock_exists, mock_get_config
    ):
        """Test _load_csv_data when CSV file exists"""
        # Setup mocks
        mock_config = Mock()
        mock_config.local_sql_db.database_path = "/tmp/test.db"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_csv_path = "/tmp/test.csv"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_database_name = "test_table"
        mock_get_config.return_value = mock_config

        mock_exists.side_effect = lambda path: path in ["/tmp", "/tmp/test.csv"]  # nosec B108: acceptable for testing

        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None
        mock_connect.return_value = mock_connection

        # Mock DataFrame with to_sql method
        mock_df = Mock(spec=pd.DataFrame)
        mock_df.to_sql = Mock()
        mock_df.dtypes = pd.Series({"name": "object", "age": "int64"})
        mock_read_csv.return_value = mock_df

        # Initialize
        with patch("builtins.print") as mock_print:
            sqlite_sample_db()

            # Verify CSV data loading
            mock_df.to_sql.assert_called_with(
                "test_table", mock_connection, if_exists="replace", index=False
            )
            mock_print.assert_any_call("CSV data loaded into test_table table.")

    @patch("ingenious.utils.load_sample_data.get_config")
    @patch("ingenious.utils.load_sample_data.os.path.exists")
    @patch("ingenious.utils.load_sample_data.sqlite3.connect")
    def test_load_csv_data_file_not_found(
        self, mock_connect, mock_exists, mock_get_config
    ):
        """Test _load_csv_data when CSV file doesn't exist"""
        # Setup mocks
        mock_config = Mock()
        mock_config.local_sql_db.database_path = "/tmp/test.db"  # nosec B108: acceptable for testing
        mock_config.local_sql_db.sample_csv_path = "/tmp/nonexistent.csv"  # nosec B108: acceptable for testing
        mock_get_config.return_value = mock_config

        mock_exists.side_effect = (
            lambda path: path == "/tmp"
        )  # Only directory exists  # nosec B108: acceptable for testing

        mock_connection = MagicMock()
        mock_connection.__enter__.return_value = mock_connection
        mock_connection.__exit__.return_value = None
        mock_connect.return_value = mock_connection

        # Initialize
        with patch("builtins.print") as mock_print:
            sqlite_sample_db()

            # Verify error message
            mock_print.assert_any_call("CSV file not found at /tmp/nonexistent.csv.")  # nosec B108: acceptable for testing

    def test_execute_sql_with_parameters(self):
        """Test execute_sql method with parameters"""
        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_file:
            db_path = temp_file.name

        try:
            # Setup test data
            conn = sqlite3.connect(db_path)
            conn.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob')")
            conn.commit()
            conn.close()

            # Create instance with minimal mocking
            with patch(
                "ingenious.utils.load_sample_data.get_config"
            ) as mock_get_config:
                mock_config = Mock()
                mock_config.local_sql_db.database_path = "/tmp/dummy.db"  # nosec B108: acceptable for testing
                mock_config.local_sql_db.sample_csv_path = "/nonexistent.csv"
                mock_get_config.return_value = mock_config

                with patch(
                    "ingenious.utils.load_sample_data.os.path.exists",
                    return_value=False,
                ):
                    with patch(
                        "ingenious.utils.load_sample_data.sqlite3.connect"
                    ) as mock_connect:
                        mock_connection = MagicMock()
                        mock_connection.__enter__.return_value = mock_connection
                        mock_connection.__exit__.return_value = None
                        mock_connect.return_value = mock_connection
                        db = sqlite_sample_db()

                # Override db_path for the actual test
                db.db_path = db_path
                result = db.execute_sql("SELECT * FROM test WHERE id = ?", [1])

                assert result is not None
                assert len(result) == 1
                assert result[0]["id"] == 1
                assert result[0]["name"] == "Alice"

        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
