from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Dialect(ABC):
    """Abstract base class for database dialects."""

    @abstractmethod
    def get_create_table_if_not_exists_prefix(self) -> str:
        """Return the database-specific prefix for CREATE TABLE IF NOT EXISTS."""
        pass

    @abstractmethod
    def get_limit_clause(self, limit: int) -> str:
        """Return the database-specific LIMIT clause."""
        pass

    @abstractmethod
    def get_upsert_query(
        self, table: str, columns: List[str], conflict_column: str
    ) -> str:
        """Return the database-specific UPSERT query."""
        pass

    @abstractmethod
    def get_temp_table_syntax(self, table_name: str, select_query: str) -> str:
        """Return the database-specific temporary table creation syntax."""
        pass

    @abstractmethod
    def get_drop_temp_table_syntax(self, table_name: str) -> str:
        """Return the database-specific temporary table drop syntax."""
        pass

    @abstractmethod
    def get_data_types(self) -> Dict[str, str]:
        """Return mapping of generic data types to database-specific types."""
        pass


class SQLiteDialect(Dialect):
    """SQLite-specific dialect implementation."""

    def get_create_table_if_not_exists_prefix(self) -> str:
        return "CREATE TABLE IF NOT EXISTS"

    def get_limit_clause(self, limit: int) -> str:
        return f"LIMIT {limit}"

    def get_upsert_query(
        self, table: str, columns: List[str], conflict_column: str
    ) -> str:
        columns_str = ", ".join(f'"{col}"' for col in columns)
        values_str = ", ".join("?" for _ in columns)
        updates_str = ", ".join(
            f'"{col}" = EXCLUDED."{col}"' for col in columns if col != conflict_column
        )

        # nosec B608: table name validated by caller, parameters use ? placeholders
        return f"""
            INSERT INTO {table} ({columns_str})
            VALUES ({values_str})
            ON CONFLICT ("{conflict_column}") DO UPDATE
            SET {updates_str}
        """

    def get_temp_table_syntax(self, table_name: str, select_query: str) -> str:
        # nosec B608: table_name validated by caller, select_query constructed internally
        return f"""
            CREATE TEMP TABLE {table_name} AS
            {select_query}
        """

    def get_drop_temp_table_syntax(self, table_name: str) -> str:
        # nosec B608: table_name validated by caller
        return f"DROP TABLE {table_name}"

    def get_data_types(self) -> Dict[str, str]:
        return {
            "uuid": "UUID",
            "varchar": "TEXT",
            "text": "TEXT",
            "boolean": "BOOLEAN",
            "datetime": "TEXT",
            "int": "INT",
            "json": "JSONB",
            "array": "TEXT[]",
        }


class AzureSQLDialect(Dialect):
    """Azure SQL-specific dialect implementation."""

    def get_create_table_if_not_exists_prefix(self) -> str:
        return "IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='{table_name}' AND xtype='U')\nCREATE TABLE"

    def get_limit_clause(self, limit: int) -> str:
        return f"TOP {limit}"

    def get_upsert_query(
        self, table: str, columns: List[str], conflict_column: str
    ) -> str:
        columns_str = ", ".join(f"[{col}]" for col in columns)
        values_str = ", ".join("?" for _ in columns)
        updates_str = ", ".join(
            f"[{col}] = ?" for col in columns if col != conflict_column
        )

        # nosec B608: table name validated by caller, parameters use ? placeholders
        return f"""
            MERGE {table} AS target
            USING (SELECT ? as {conflict_column}) AS source ON target.[{conflict_column}] = source.{conflict_column}
            WHEN MATCHED THEN
                UPDATE SET {updates_str}
            WHEN NOT MATCHED THEN
                INSERT ({columns_str})
                VALUES ({values_str})
        """

    def get_temp_table_syntax(self, table_name: str, select_query: str) -> str:
        # nosec B608: table name validated by caller, select_query is constructed internally
        return f"""
            {select_query}
            INTO #{table_name}
        """

    def get_drop_temp_table_syntax(self, table_name: str) -> str:
        # nosec B608: table_name validated by caller
        return f"DROP TABLE #{table_name}"

    def get_data_types(self) -> Dict[str, str]:
        return {
            "uuid": "UNIQUEIDENTIFIER",
            "varchar": "NVARCHAR(255)",
            "text": "NVARCHAR(MAX)",
            "boolean": "BIT",
            "datetime": "DATETIME2",
            "int": "INT",
            "json": "NVARCHAR(MAX)",
            "array": "NVARCHAR(MAX)",
        }


class QueryBuilder:
    """Centralized query builder that generates database-specific SQL queries."""

    def __init__(self, dialect: Dialect) -> None:
        self.dialect = dialect
        self._data_types = dialect.get_data_types()

    def _get_data_type(self, generic_type: str) -> str:
        """Get database-specific data type."""
        return self._data_types.get(generic_type, generic_type)

    def create_chat_history_table(self) -> str:
        """Generate CREATE TABLE query for chat_history."""
        table_name = "chat_history"
        prefix = self.dialect.get_create_table_if_not_exists_prefix()
        if "{table_name}" in prefix:
            prefix = prefix.format(table_name=table_name)

        # nosec B608: table name 'chat_history' is hardcoded constant, parameters use ? placeholders
        return f"""
            {prefix} {table_name} (
                user_id {self._get_data_type("varchar")},
                thread_id {self._get_data_type("varchar")},
                message_id {self._get_data_type("varchar")},
                positive_feedback {self._get_data_type("boolean")},
                timestamp {self._get_data_type("datetime")},
                role {self._get_data_type("varchar")},
                content {self._get_data_type("text")},
                content_filter_results {self._get_data_type("text")},
                tool_calls {self._get_data_type("text")},
                tool_call_id {self._get_data_type("varchar")},
                tool_call_function {self._get_data_type("varchar")}
            );
        """

    def create_chat_history_summary_table(self) -> str:
        """Generate CREATE TABLE query for chat_history_summary."""
        table_name = "chat_history_summary"
        prefix = self.dialect.get_create_table_if_not_exists_prefix()
        if "{table_name}" in prefix:
            prefix = prefix.format(table_name=table_name)

        # nosec B608: table name 'chat_history_summary' is hardcoded constant, parameters use ? placeholders
        return f"""
            {prefix} {table_name} (
                user_id {self._get_data_type("varchar")},
                thread_id {self._get_data_type("varchar")},
                message_id {self._get_data_type("varchar")},
                positive_feedback {self._get_data_type("boolean")},
                timestamp {self._get_data_type("datetime")},
                role {self._get_data_type("varchar")},
                content {self._get_data_type("text")},
                content_filter_results {self._get_data_type("text")},
                tool_calls {self._get_data_type("text")},
                tool_call_id {self._get_data_type("varchar")},
                tool_call_function {self._get_data_type("varchar")}
            );
        """

    def create_users_table(self) -> str:
        """Generate CREATE TABLE query for users."""
        table_name = "users"
        prefix = self.dialect.get_create_table_if_not_exists_prefix()
        if "{table_name}" in prefix:
            prefix = prefix.format(table_name=table_name)

        # nosec B608: table name 'users' is hardcoded constant, parameters use ? placeholders
        return f"""
            {prefix} {table_name} (
                id {self._get_data_type("uuid")} PRIMARY KEY,
                identifier {self._get_data_type("varchar")} NOT NULL UNIQUE,
                metadata {self._get_data_type("json")} NOT NULL,
                createdAt {self._get_data_type("datetime")}
            );
        """

    def create_threads_table(self) -> str:
        """Generate CREATE TABLE query for threads."""
        table_name = "threads"
        prefix = self.dialect.get_create_table_if_not_exists_prefix()
        if "{table_name}" in prefix:
            prefix = prefix.format(table_name=table_name)

        foreign_key = ""
        if isinstance(self.dialect, AzureSQLDialect):
            foreign_key = "FOREIGN KEY (userId) REFERENCES users(id) ON DELETE CASCADE"
        else:
            foreign_key = (
                'FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE'
            )

        # nosec B608: table name 'threads' is hardcoded constant, parameters use ? placeholders
        return f"""
            {prefix} {table_name} (
                id {self._get_data_type("uuid")} PRIMARY KEY,
                createdAt {self._get_data_type("datetime")},
                name {self._get_data_type("varchar")},
                userId {self._get_data_type("uuid")},
                userIdentifier {self._get_data_type("varchar")},
                tags {self._get_data_type("array")},
                metadata {self._get_data_type("json")},
                {foreign_key}
            );
        """

    def create_steps_table(self) -> str:
        """Generate CREATE TABLE query for steps."""
        table_name = "steps"
        prefix = self.dialect.get_create_table_if_not_exists_prefix()
        if "{table_name}" in prefix:
            prefix = prefix.format(table_name=table_name)

        # Handle 'end' column name conflict in SQL Server
        end_column = "[end]" if isinstance(self.dialect, AzureSQLDialect) else "end"

        # nosec B608: table name 'steps' is hardcoded constant, parameters use ? placeholders
        return f"""
            {prefix} {table_name} (
                id {self._get_data_type("uuid")} PRIMARY KEY,
                name {self._get_data_type("varchar")} NOT NULL,
                type {self._get_data_type("varchar")} NOT NULL,
                threadId {self._get_data_type("uuid")} NOT NULL,
                parentId {self._get_data_type("uuid")},
                disableFeedback {self._get_data_type("boolean")} NOT NULL,
                streaming {self._get_data_type("boolean")} NOT NULL,
                waitForAnswer {self._get_data_type("boolean")},
                isError {self._get_data_type("boolean")},
                metadata {self._get_data_type("json")},
                tags {self._get_data_type("array")},
                input {self._get_data_type("text")},
                output {self._get_data_type("text")},
                createdAt {self._get_data_type("datetime")},
                start {self._get_data_type("datetime")},
                {end_column} {self._get_data_type("datetime")},
                generation {self._get_data_type("json")},
                showInput {self._get_data_type("varchar")},
                language {self._get_data_type("varchar")},
                indent {self._get_data_type("int")}
            );
        """

    def create_elements_table(self) -> str:
        """Generate CREATE TABLE query for elements."""
        table_name = "elements"
        prefix = self.dialect.get_create_table_if_not_exists_prefix()
        if "{table_name}" in prefix:
            prefix = prefix.format(table_name=table_name)

        # nosec B608: table name 'elements' is hardcoded constant, parameters use ? placeholders
        return f"""
            {prefix} {table_name} (
                id {self._get_data_type("uuid")} PRIMARY KEY,
                threadId {self._get_data_type("uuid")},
                type {self._get_data_type("varchar")},
                url {self._get_data_type("text")},
                chainlitKey {self._get_data_type("varchar")},
                name {self._get_data_type("varchar")} NOT NULL,
                display {self._get_data_type("varchar")},
                objectKey {self._get_data_type("varchar")},
                size {self._get_data_type("varchar")},
                page {self._get_data_type("int")},
                language {self._get_data_type("varchar")},
                forId {self._get_data_type("uuid")},
                mime {self._get_data_type("varchar")}
            );
        """

    def create_feedbacks_table(self) -> str:
        """Generate CREATE TABLE query for feedbacks."""
        table_name = "feedbacks"
        prefix = self.dialect.get_create_table_if_not_exists_prefix()
        if "{table_name}" in prefix:
            prefix = prefix.format(table_name=table_name)

        # nosec B608: table name 'feedbacks' is hardcoded constant, parameters use ? placeholders
        return f"""
            {prefix} {table_name} (
                id {self._get_data_type("uuid")} PRIMARY KEY,
                forId {self._get_data_type("uuid")} NOT NULL,
                threadId {self._get_data_type("uuid")} NOT NULL,
                value {self._get_data_type("int")} NOT NULL,
                comment {self._get_data_type("text")}
            );
        """

    def insert_message(self) -> str:
        """Generate INSERT query for messages."""
        return """
            INSERT INTO chat_history (
                user_id, thread_id, message_id, positive_feedback, timestamp,
                role, content, content_filter_results, tool_calls,
                tool_call_id, tool_call_function)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

    def insert_memory(self) -> str:
        """Generate INSERT query for memory."""
        return """
            INSERT INTO chat_history_summary (
                user_id, thread_id, message_id, positive_feedback, timestamp,
                role, content, content_filter_results, tool_calls,
                tool_call_id, tool_call_function)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

    def select_message(self) -> str:
        """Generate SELECT query for a specific message."""
        return """
            SELECT user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                   content_filter_results, tool_calls, tool_call_id, tool_call_function
            FROM chat_history
            WHERE message_id = ? AND thread_id = ?
        """

    def select_latest_memory(self) -> str:
        """Generate SELECT query for latest memory."""
        limit_clause = self.dialect.get_limit_clause(1)

        if isinstance(self.dialect, AzureSQLDialect):
            # nosec B608: table name 'chat_history_summary' is hardcoded constant, parameters use ? placeholders
            return f"""
                SELECT {limit_clause} user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                       content_filter_results, tool_calls, tool_call_id, tool_call_function
                FROM chat_history_summary
                WHERE thread_id = ?
                ORDER BY timestamp DESC
            """
        else:
            # nosec B608: table name 'chat_history_summary' is hardcoded constant, parameters use ? placeholders
            return f"""
                SELECT user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                       content_filter_results, tool_calls, tool_call_id, tool_call_function
                FROM chat_history_summary
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                {limit_clause}
            """

    def update_message_feedback(self) -> str:
        """Generate UPDATE query for message feedback."""
        return """
            UPDATE chat_history
            SET positive_feedback = ?
            WHERE message_id = ? AND thread_id = ?
        """

    def update_memory_feedback(self) -> str:
        """Generate UPDATE query for memory feedback."""
        return """
            UPDATE chat_history_summary
            SET positive_feedback = ?
            WHERE message_id = ? AND thread_id = ?
        """

    def update_message_content_filter(self) -> str:
        """Generate UPDATE query for message content filter results."""
        return """
            UPDATE chat_history
            SET content_filter_results = ?
            WHERE message_id = ? AND thread_id = ?
        """

    def update_memory_content_filter(self) -> str:
        """Generate UPDATE query for memory content filter results."""
        return """
            UPDATE chat_history_summary
            SET content_filter_results = ?
            WHERE message_id = ? AND thread_id = ?
        """

    def insert_user(self) -> str:
        """Generate INSERT query for users."""
        return """
            INSERT INTO users (id, identifier, metadata, createdAt)
            VALUES (?, ?, ?, ?)
        """

    def select_user(self) -> str:
        """Generate SELECT query for users."""
        return """
            SELECT id, identifier, metadata, createdAt
            FROM users
            WHERE identifier = ?
        """

    def select_thread_messages(self, limit: int = 5) -> str:
        """Generate SELECT query for thread messages."""
        if isinstance(self.dialect, AzureSQLDialect):
            # nosec B608: table name 'chat_history' is hardcoded constant, parameters use ? placeholders
            return f"""
                SELECT TOP {limit} user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                       content_filter_results, tool_calls, tool_call_id, tool_call_function
                FROM (
                    SELECT user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                           content_filter_results, tool_calls, tool_call_id, tool_call_function,
                           ROW_NUMBER() OVER (ORDER BY timestamp DESC) as rn
                    FROM chat_history
                    WHERE thread_id = ?
                ) AS ranked
                WHERE rn <= {limit}
                ORDER BY timestamp ASC
            """
        else:
            # nosec B608: table name 'chat_history' is hardcoded constant, parameters use ? placeholders
            return f"""
                SELECT *
                FROM (
                    SELECT user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                           content_filter_results, tool_calls, tool_call_id, tool_call_function
                    FROM chat_history
                    WHERE thread_id = ?
                    ORDER BY timestamp DESC
                    LIMIT {limit}
                ) AS last_five
                ORDER BY timestamp ASC
            """

    def select_thread_memory(self) -> str:
        """Generate SELECT query for thread memory."""
        limit_clause = self.dialect.get_limit_clause(1)

        if isinstance(self.dialect, AzureSQLDialect):
            # nosec B608: table name 'chat_history_summary' is hardcoded constant, parameters use ? placeholders
            return f"""
                SELECT {limit_clause} user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                       content_filter_results, tool_calls, tool_call_id, tool_call_function
                FROM chat_history_summary
                WHERE thread_id = ?
                ORDER BY timestamp DESC
            """
        else:
            # nosec B608: table name 'chat_history_summary' is hardcoded constant, parameters use ? placeholders
            return f"""
                SELECT user_id, thread_id, message_id, positive_feedback, timestamp, role, content,
                       content_filter_results, tool_calls, tool_call_id, tool_call_function
                FROM chat_history_summary
                WHERE thread_id = ?
                ORDER BY timestamp DESC
                {limit_clause}
            """

    def delete_thread(self) -> str:
        """Generate DELETE query for thread messages."""
        return """
            DELETE FROM chat_history
            WHERE thread_id = ?
        """

    def delete_thread_memory(self) -> str:
        """Generate DELETE query for thread memory."""
        return """
            DELETE FROM chat_history_summary
            WHERE thread_id = ?
        """

    def delete_user_memory(self) -> str:
        """Generate DELETE query for user memory."""
        return """
            DELETE FROM chat_history_summary
            WHERE user_id = ?
        """

    def get_query(self, query_type: str, **kwargs: Any) -> str:
        """Get a query by type with optional parameters."""
        method_name = query_type
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            if callable(method):
                result = method(**kwargs)
                return str(result) if result is not None else ""
        return ""
