"""
Tests for database query builder functionality.

This test suite covers SQL query generation for different database dialects
including SQLite and Azure SQL Server.
"""

import pytest

from ingenious.db.query_builder import (
    AzureSQLDialect,
    Dialect,
    QueryBuilder,
    SQLiteDialect,
)


class TestSQLiteDialect:
    """Test SQLite-specific dialect implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dialect = SQLiteDialect()

    def test_get_create_table_if_not_exists_prefix(self):
        """Test CREATE TABLE IF NOT EXISTS prefix for SQLite."""
        result = self.dialect.get_create_table_if_not_exists_prefix()
        assert result == "CREATE TABLE IF NOT EXISTS"

    def test_get_limit_clause(self):
        """Test LIMIT clause generation for SQLite."""
        result = self.dialect.get_limit_clause(10)
        assert result == "LIMIT 10"

        result = self.dialect.get_limit_clause(1)
        assert result == "LIMIT 1"

    def test_get_upsert_query(self):
        """Test UPSERT query generation for SQLite."""
        table = "test_table"
        columns = ["id", "name", "value"]
        conflict_column = "id"

        result = self.dialect.get_upsert_query(table, columns, conflict_column)

        assert "INSERT INTO test_table" in result
        assert '"id", "name", "value"' in result
        assert "?, ?, ?" in result
        assert "ON CONFLICT" in result
        assert 'EXCLUDED."name"' in result
        assert 'EXCLUDED."value"' in result
        # Should not update the conflict column
        assert 'EXCLUDED."id"' not in result

    def test_get_upsert_query_single_column(self):
        """Test UPSERT query with single non-conflict column."""
        result = self.dialect.get_upsert_query("test", ["id", "name"], "id")
        assert 'SET "name" = EXCLUDED."name"' in result

    def test_get_temp_table_syntax(self):
        """Test temporary table creation syntax for SQLite."""
        table_name = "temp_test"
        select_query = "SELECT * FROM source_table"

        result = self.dialect.get_temp_table_syntax(table_name, select_query)

        assert "CREATE TEMP TABLE temp_test AS" in result
        assert "SELECT * FROM source_table" in result

    def test_get_drop_temp_table_syntax(self):
        """Test temporary table drop syntax for SQLite."""
        result = self.dialect.get_drop_temp_table_syntax("temp_test")
        assert result == "DROP TABLE temp_test"

    def test_get_data_types(self):
        """Test SQLite data type mappings."""
        data_types = self.dialect.get_data_types()

        expected_types = {
            "uuid": "UUID",
            "varchar": "TEXT",
            "text": "TEXT",
            "boolean": "BOOLEAN",
            "datetime": "TEXT",
            "int": "INT",
            "json": "JSONB",
            "array": "TEXT[]",
        }

        assert data_types == expected_types


class TestAzureSQLDialect:
    """Test Azure SQL-specific dialect implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.dialect = AzureSQLDialect()

    def test_get_create_table_if_not_exists_prefix(self):
        """Test CREATE TABLE IF NOT EXISTS prefix for Azure SQL."""
        result = self.dialect.get_create_table_if_not_exists_prefix()
        expected = "IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')\nCREATE TABLE"
        assert result == expected

    def test_get_limit_clause(self):
        """Test LIMIT clause generation for Azure SQL (uses TOP)."""
        result = self.dialect.get_limit_clause(10)
        assert result == "TOP 10"

        result = self.dialect.get_limit_clause(5)
        assert result == "TOP 5"

    def test_get_upsert_query(self):
        """Test UPSERT query generation for Azure SQL (uses MERGE)."""
        table = "test_table"
        columns = ["id", "name", "value"]
        conflict_column = "id"

        result = self.dialect.get_upsert_query(table, columns, conflict_column)

        assert "MERGE test_table AS target" in result
        assert "[id], [name], [value]" in result
        assert "WHEN MATCHED THEN" in result
        assert "WHEN NOT MATCHED THEN" in result
        assert "[name] = ?" in result
        assert "[value] = ?" in result
        # Should not update the conflict column
        assert "[id] = ?" not in result

    def test_get_temp_table_syntax(self):
        """Test temporary table creation syntax for Azure SQL."""
        table_name = "temp_test"
        select_query = "SELECT * FROM source_table"

        result = self.dialect.get_temp_table_syntax(table_name, select_query)

        assert "SELECT * FROM source_table" in result
        assert "INTO #temp_test" in result

    def test_get_drop_temp_table_syntax(self):
        """Test temporary table drop syntax for Azure SQL."""
        result = self.dialect.get_drop_temp_table_syntax("temp_test")
        assert result == "DROP TABLE #temp_test"

    def test_get_data_types(self):
        """Test Azure SQL data type mappings."""
        data_types = self.dialect.get_data_types()

        expected_types = {
            "uuid": "UNIQUEIDENTIFIER",
            "varchar": "NVARCHAR(255)",
            "text": "NVARCHAR(MAX)",
            "boolean": "BIT",
            "datetime": "DATETIME2",
            "int": "INT",
            "json": "NVARCHAR(MAX)",
            "array": "NVARCHAR(MAX)",
        }

        assert data_types == expected_types


class TestQueryBuilderWithSQLite:
    """Test QueryBuilder functionality with SQLite dialect."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = QueryBuilder(SQLiteDialect())

    def test_init(self):
        """Test QueryBuilder initialization."""
        assert isinstance(self.builder.dialect, SQLiteDialect)
        assert self.builder._data_types is not None

    def test_get_data_type(self):
        """Test data type resolution."""
        assert self.builder._get_data_type("uuid") == "UUID"
        assert self.builder._get_data_type("varchar") == "TEXT"
        assert self.builder._get_data_type("unknown_type") == "unknown_type"  # fallback

    def test_create_chat_history_table(self):
        """Test chat history table creation query."""
        query = self.builder.create_chat_history_table()

        assert "CREATE TABLE IF NOT EXISTS chat_history" in query
        assert "user_id TEXT" in query
        assert "thread_id TEXT" in query
        assert "message_id TEXT" in query
        assert "positive_feedback BOOLEAN" in query
        assert "timestamp TEXT" in query
        assert "role TEXT" in query
        assert "content TEXT" in query

    def test_create_chat_history_summary_table(self):
        """Test chat history summary table creation query."""
        query = self.builder.create_chat_history_summary_table()

        assert "CREATE TABLE IF NOT EXISTS chat_history_summary" in query
        assert "user_id TEXT" in query
        assert "thread_id TEXT" in query

    def test_create_users_table(self):
        """Test users table creation query."""
        query = self.builder.create_users_table()

        assert "CREATE TABLE IF NOT EXISTS users" in query
        assert "id UUID PRIMARY KEY" in query
        assert "identifier TEXT NOT NULL UNIQUE" in query
        assert "metadata JSONB NOT NULL" in query

    def test_create_threads_table(self):
        """Test threads table creation query."""
        query = self.builder.create_threads_table()

        assert "CREATE TABLE IF NOT EXISTS threads" in query
        assert "id UUID PRIMARY KEY" in query
        assert "userId UUID" in query
        assert "FOREIGN KEY" in query
        assert "ON DELETE CASCADE" in query

    def test_create_steps_table(self):
        """Test steps table creation query."""
        query = self.builder.create_steps_table()

        assert "CREATE TABLE IF NOT EXISTS steps" in query
        assert "id UUID PRIMARY KEY" in query
        assert "name TEXT NOT NULL" in query
        assert "type TEXT NOT NULL" in query
        assert "end TEXT" in query  # SQLite doesn't need brackets

    def test_create_elements_table(self):
        """Test elements table creation query."""
        query = self.builder.create_elements_table()

        assert "CREATE TABLE IF NOT EXISTS elements" in query
        assert "id UUID PRIMARY KEY" in query
        assert "threadId UUID" in query
        assert "type TEXT" in query

    def test_create_feedbacks_table(self):
        """Test feedbacks table creation query."""
        query = self.builder.create_feedbacks_table()

        assert "CREATE TABLE IF NOT EXISTS feedbacks" in query
        assert "id UUID PRIMARY KEY" in query
        assert "forId UUID NOT NULL" in query
        assert "value INT NOT NULL" in query

    def test_insert_message(self):
        """Test message insertion query."""
        query = self.builder.insert_message()

        assert "INSERT INTO chat_history" in query
        assert "user_id, thread_id, message_id" in query
        assert "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)" in query

    def test_insert_memory(self):
        """Test memory insertion query."""
        query = self.builder.insert_memory()

        assert "INSERT INTO chat_history_summary" in query
        assert "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)" in query

    def test_select_message(self):
        """Test message selection query."""
        query = self.builder.select_message()

        assert "SELECT user_id, thread_id, message_id" in query
        assert "FROM chat_history" in query
        assert "WHERE message_id = ? AND thread_id = ?" in query

    def test_select_latest_memory(self):
        """Test latest memory selection query."""
        query = self.builder.select_latest_memory()

        assert "SELECT user_id, thread_id, message_id" in query
        assert "FROM chat_history_summary" in query
        assert "WHERE thread_id = ?" in query
        assert "ORDER BY timestamp DESC" in query
        assert "LIMIT 1" in query

    def test_update_message_feedback(self):
        """Test message feedback update query."""
        query = self.builder.update_message_feedback()

        assert "UPDATE chat_history" in query
        assert "SET positive_feedback = ?" in query
        assert "WHERE message_id = ? AND thread_id = ?" in query

    def test_update_memory_feedback(self):
        """Test memory feedback update query."""
        query = self.builder.update_memory_feedback()

        assert "UPDATE chat_history_summary" in query
        assert "SET positive_feedback = ?" in query

    def test_update_message_content_filter(self):
        """Test message content filter update query."""
        query = self.builder.update_message_content_filter()

        assert "UPDATE chat_history" in query
        assert "SET content_filter_results = ?" in query

    def test_update_memory_content_filter(self):
        """Test memory content filter update query."""
        query = self.builder.update_memory_content_filter()

        assert "UPDATE chat_history_summary" in query
        assert "SET content_filter_results = ?" in query

    def test_insert_user(self):
        """Test user insertion query."""
        query = self.builder.insert_user()

        assert "INSERT INTO users" in query
        assert "VALUES (?, ?, ?, ?)" in query

    def test_select_user(self):
        """Test user selection query."""
        query = self.builder.select_user()

        assert "SELECT id, identifier, metadata, createdAt" in query
        assert "FROM users" in query
        assert "WHERE identifier = ?" in query

    def test_select_thread_messages_default_limit(self):
        """Test thread messages selection with default limit."""
        query = self.builder.select_thread_messages()

        assert "SELECT *" in query
        assert "FROM chat_history" in query
        assert "WHERE thread_id = ?" in query
        assert "LIMIT 5" in query
        assert "ORDER BY timestamp" in query

    def test_select_thread_messages_custom_limit(self):
        """Test thread messages selection with custom limit."""
        query = self.builder.select_thread_messages(limit=10)

        assert "LIMIT 10" in query

    def test_select_thread_memory(self):
        """Test thread memory selection query."""
        query = self.builder.select_thread_memory()

        assert "SELECT user_id, thread_id, message_id" in query
        assert "FROM chat_history_summary" in query
        assert "WHERE thread_id = ?" in query
        assert "LIMIT 1" in query

    def test_delete_thread(self):
        """Test thread deletion query."""
        query = self.builder.delete_thread()

        assert "DELETE FROM chat_history" in query
        assert "WHERE thread_id = ?" in query

    def test_delete_thread_memory(self):
        """Test thread memory deletion query."""
        query = self.builder.delete_thread_memory()

        assert "DELETE FROM chat_history_summary" in query
        assert "WHERE thread_id = ?" in query

    def test_delete_user_memory(self):
        """Test user memory deletion query."""
        query = self.builder.delete_user_memory()

        assert "DELETE FROM chat_history_summary" in query
        assert "WHERE user_id = ?" in query


class TestQueryBuilderWithAzureSQL:
    """Test QueryBuilder functionality with Azure SQL dialect."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = QueryBuilder(AzureSQLDialect())

    def test_create_chat_history_table_azure(self):
        """Test chat history table creation query for Azure SQL."""
        query = self.builder.create_chat_history_table()

        assert "IF NOT EXISTS" in query
        assert "sysobjects" in query
        assert "user_id NVARCHAR(255)" in query
        assert "positive_feedback BIT" in query
        assert "timestamp DATETIME2" in query

    def test_create_threads_table_azure(self):
        """Test threads table creation query for Azure SQL."""
        query = self.builder.create_threads_table()

        assert "userId UNIQUEIDENTIFIER" in query
        assert (
            "FOREIGN KEY (userId) REFERENCES users(id)" in query
        )  # No quotes in Azure SQL

    def test_create_steps_table_azure(self):
        """Test steps table creation query for Azure SQL."""
        query = self.builder.create_steps_table()

        assert "[end] DATETIME2" in query  # Azure SQL uses brackets for reserved words

    def test_select_latest_memory_azure(self):
        """Test latest memory selection query for Azure SQL."""
        query = self.builder.select_latest_memory()

        assert "SELECT TOP 1" in query  # Azure SQL uses TOP instead of LIMIT
        assert "ORDER BY timestamp DESC" in query

    def test_select_thread_messages_azure(self):
        """Test thread messages selection query for Azure SQL."""
        query = self.builder.select_thread_messages(limit=3)

        assert "SELECT TOP 3" in query
        assert "ROW_NUMBER() OVER" in query
        assert "WHERE rn <= 3" in query

    def test_select_thread_memory_azure(self):
        """Test thread memory selection query for Azure SQL."""
        query = self.builder.select_thread_memory()

        assert "SELECT TOP 1" in query
        assert "ORDER BY timestamp DESC" in query


class TestQueryBuilderGeneral:
    """Test general QueryBuilder functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.builder = QueryBuilder(SQLiteDialect())

    def test_get_query_with_existing_method(self):
        """Test get_query with existing method."""
        result = self.builder.get_query("insert_message")
        assert "INSERT INTO chat_history" in result

    def test_get_query_with_parameters(self):
        """Test get_query with method parameters."""
        result = self.builder.get_query("select_thread_messages", limit=10)
        assert "LIMIT 10" in result

    def test_get_query_with_nonexistent_method(self):
        """Test get_query with non-existent method."""
        result = self.builder.get_query("nonexistent_method")
        assert result == ""

    def test_get_query_with_non_callable_attribute(self):
        """Test get_query with non-callable attribute."""
        # Add a non-callable attribute
        self.builder.test_attr = "not_callable"
        result = self.builder.get_query("test_attr")
        assert result == ""


class TestDialectAbstract:
    """Test that Dialect abstract base class cannot be instantiated."""

    def test_cannot_instantiate_abstract_dialect(self):
        """Test that abstract Dialect cannot be instantiated."""
        with pytest.raises(TypeError):
            Dialect()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_upsert_query_all_columns_conflict(self):
        """Test UPSERT when conflict column is the only column."""
        sqlite_dialect = SQLiteDialect()
        result = sqlite_dialect.get_upsert_query("test", ["id"], "id")

        # Should have empty SET clause when no non-conflict columns
        assert "SET " not in result or "SET \n" in result

    def test_azure_sql_table_name_formatting(self):
        """Test Azure SQL table name formatting in CREATE TABLE prefix."""
        builder = QueryBuilder(AzureSQLDialect())
        query = builder.create_users_table()

        # The prefix should have {table_name} replaced
        assert "{table_name}" not in query
        assert "users" in query

    def test_query_builder_different_data_types(self):
        """Test QueryBuilder with different data type scenarios."""
        builder = QueryBuilder(SQLiteDialect())

        # Test that unknown data types fall back to themselves
        unknown_type = builder._get_data_type("custom_type")
        assert unknown_type == "custom_type"

    def test_empty_columns_in_upsert(self):
        """Test UPSERT with empty columns list."""
        sqlite_dialect = SQLiteDialect()
        result = sqlite_dialect.get_upsert_query("test", [], "id")

        # Should handle empty columns gracefully
        assert "test" in result
        assert "INSERT INTO" in result
