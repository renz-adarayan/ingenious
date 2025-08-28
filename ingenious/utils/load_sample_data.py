import os
import sqlite3
from typing import Any, Dict, List, Optional

import pandas as pd  # Import pandas for CSV handling

from ingenious.config.config import get_config


class sqlite_sample_db:
    def __init__(self) -> None:
        self._config = get_config()

        self.db_path: str = self._config.local_sql_db.database_path
        db_dir_check: str = os.path.dirname(self.db_path)
        if not os.path.exists(db_dir_check):
            os.makedirs(db_dir_check, exist_ok=True)
        self.connection: sqlite3.Connection = sqlite3.connect(
            self.db_path, check_same_thread=False
        )
        self._create_table()
        self._load_csv_data()

    def execute_sql(
        self, sql: str, params: List[Any] = [], expect_results: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        connection: Optional[sqlite3.Connection] = None
        try:
            connection = sqlite3.connect(self.db_path)
            connection.row_factory = sqlite3.Row  # Set row factory to sqlite3.Row
            cursor: sqlite3.Cursor = connection.cursor()

            if expect_results:
                res: sqlite3.Cursor = cursor.execute(sql, params)
                rows: List[sqlite3.Row] = res.fetchall()
                result: List[Dict[str, Any]] = [dict(row) for row in rows]
                return result
            else:
                connection.execute(sql, params)
                connection.commit()
                return None

        except sqlite3.Error as e:
            # Display the exception
            print(e)
            return None

        finally:
            if connection:
                connection.close()

    def _create_table(self) -> None:
        # Dynamic table creation based on CSV structure
        csv_path: str = self._config.local_sql_db.sample_csv_path
        if os.path.exists(csv_path):
            df: pd.DataFrame = pd.read_csv(csv_path)
            # Infer SQL types from pandas dtypes
            column_definitions: List[str] = []
            for col, dtype in df.dtypes.items():
                if dtype == "int64":
                    column_definitions.append(f"{col} INTEGER")
                elif dtype == "float64":
                    column_definitions.append(f"{col} REAL")
                else:
                    column_definitions.append(f"{col} TEXT")

            table_name: str = self._config.local_sql_db.sample_database_name
            # nosec B608: table name from config, column definitions are validated CSV headers
            create_sql: str = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    {", ".join(column_definitions)}
                );
            """

            with self.connection:
                self.connection.execute(create_sql)
                print(
                    f"Created table {table_name} with columns: {', '.join(column_definitions)}"
                )
        else:
            # Fallback to hardcoded students_performance table for backwards compatibility
            with self.connection:
                self.connection.execute("""
                    CREATE TABLE IF NOT EXISTS students_performance (
                        gender TEXT,
                        race_ethnicity TEXT,
                        parental_education TEXT,
                        lunch TEXT,
                        test_prep_course TEXT,
                        math_score INTEGER,
                        reading_score INTEGER,
                        writing_score INTEGER
                    );
                """)

    def _load_csv_data(self) -> None:
        # Load CSV file into a DataFrame
        csv_path: str = self._config.local_sql_db.sample_csv_path
        if os.path.exists(csv_path):
            df: pd.DataFrame = pd.read_csv(csv_path)
            table_name: str = self._config.local_sql_db.sample_database_name
            # Load data into the table
            with self.connection:
                df.to_sql(table_name, self.connection, if_exists="replace", index=False)
            print(f"CSV data loaded into {table_name} table.")
        else:
            print(f"CSV file not found at {csv_path}.")
