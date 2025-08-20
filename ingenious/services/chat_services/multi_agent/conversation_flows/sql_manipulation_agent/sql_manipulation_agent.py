import logging
import os
import sqlite3
import uuid

from autogen_agentchat.agents import AssistantAgent
from autogen_core import EVENT_LOGGER_NAME, CancellationToken
from autogen_core.tools import FunctionTool

from ingenious.client.azure import AzureClientFactory
from ingenious.models.agent import LLMUsageTracker
from ingenious.models.chat import ChatRequest, ChatResponse
from ingenious.services.chat_services.multi_agent.service import IConversationFlow

try:
    import pyodbc

    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False


class ConversationFlow(IConversationFlow):
    async def get_conversation_response(
        self, chat_request: ChatRequest
    ) -> ChatResponse:
        # Get configuration from the parent service
        model_config = self._config.models[0]

        # Initialize LLM usage tracking
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.setLevel(logging.INFO)

        llm_logger = LLMUsageTracker(
            agents=[],  # Simple agent, no complex agent list needed
            config=self._config,
            chat_history_repository=self._chat_service.chat_history_repository
            if self._chat_service
            else None,
            revision_id=str(uuid.uuid4()),
            identifier=str(uuid.uuid4()),
            event_type="sql_manipulation",
        )

        logger.handlers = [llm_logger]

        # Retrieve thread memory for context
        memory_context = ""
        if chat_request.thread_id and self._chat_service:
            try:
                thread_messages = await self._chat_service.chat_history_repository.get_thread_messages(
                    chat_request.thread_id
                )
                if thread_messages:
                    # Build conversation context from recent messages (last 10)
                    recent_messages = (
                        thread_messages[-10:]
                        if len(thread_messages) > 10
                        else thread_messages
                    )
                    memory_parts = []
                    for msg in recent_messages:
                        memory_parts.append(f"{msg.role}: {msg.content[:100]}...")
                    memory_context = (
                        "Previous conversation:\n" + "\n".join(memory_parts) + "\n\n"
                    )
            except Exception as e:
                logger.warning(f"Failed to retrieve thread memory: {e}")

        # Create the model client
        model_client = AzureClientFactory.create_openai_chat_completion_client(
            model_config
        )

        # Set up context for conversation
        context = "SQL Expert Assistant for analyzing data."

        # Check if Azure SQL is configured
        use_azure_sql = (
            hasattr(self._config, "azure_sql_services")
            and self._config.azure_sql_services
            and PYODBC_AVAILABLE
            and self._config.azure_sql_services.database_connection_string
            and self._config.azure_sql_services.database_connection_string
            != "mock-connection-string"
        )

        if use_azure_sql:
            # Use Azure SQL configuration
            connection_string = (
                self._config.azure_sql_services.database_connection_string
            )
            table_name = self._config.azure_sql_services.table_name or "sample_table"

            # Get table schema from Azure SQL
            try:
                with pyodbc.connect(connection_string) as conn:
                    cursor = conn.cursor()
                    # Get column information
                    cursor.execute(f"""
                        SELECT COLUMN_NAME
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = '{table_name}'
                        ORDER BY ORDINAL_POSITION
                    """)
                    column_names = [row[0] for row in cursor.fetchall()]

                    if not column_names:
                        # Table doesn't exist, fallback to SQLite
                        use_azure_sql = False

            except Exception as e:
                print(f"Azure SQL connection failed, falling back to SQLite: {e}")
                use_azure_sql = False

        if not use_azure_sql:
            # Set up local SQLite database with student performance data
            # Create SQLite database with sample data
            db_path = os.path.join(self._memory_path, "students_performance.db")
            table_name = "students_performance"
            column_names = [
                "parental_education",
                "lunch",
                "test_prep_course",
                "math_score",
                "reading_score",
                "writing_score",
            ]

        if not use_azure_sql:
            # Create a simple test table with dummy data for SQLite
            with sqlite3.connect(db_path) as conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS students_performance (
                    parental_education TEXT,
                    lunch TEXT,
                    test_prep_course TEXT,
                    math_score INTEGER,
                    reading_score INTEGER,
                    writing_score INTEGER
                )""")
                # Insert some sample data if table is empty
                count = conn.execute(
                    "SELECT COUNT(*) FROM students_performance"
                ).fetchone()[0]
                if count == 0:
                    conn.execute("""INSERT INTO students_performance VALUES
                        ('bachelor''s degree', 'standard', 'none', 72, 72, 74),
                        ('some college', 'standard', 'completed', 69, 90, 88),
                        ('master''s degree', 'standard', 'none', 90, 95, 93),
                        ('associate''s degree', 'free/reduced', 'none', 47, 57, 44),
                        ('some college', 'standard', 'none', 76, 78, 75),
                        ('high school', 'free/reduced', 'completed', 64, 64, 67),
                        ('high school', 'free/reduced', 'none', 38, 60, 50)
                    """)

        # Create SQL tool as function
        async def execute_sql_tool(query: str) -> str:
            """Execute SQL query on configured database (Azure SQL or SQLite)"""
            try:
                if use_azure_sql:
                    # Execute on Azure SQL
                    with pyodbc.connect(connection_string) as conn:
                        cursor = conn.cursor()
                        cursor.execute(query)
                        results = cursor.fetchall()
                        columns = [column[0] for column in cursor.description]
                else:
                    # Execute on SQLite
                    with sqlite3.connect(db_path) as conn:
                        cursor = conn.execute(query)
                        results = cursor.fetchall()
                        columns = [description[0] for description in cursor.description]

                if not results:
                    return "No results found."

                # Format results
                if len(results) == 1:
                    # Single row: return as dictionary
                    result_dict = dict(zip(columns, results[0]))
                    return str(result_dict)
                else:
                    # Multiple rows: return as list of dictionaries
                    result_list = [dict(zip(columns, row)) for row in results]
                    return str(result_list[:10])  # Limit to first 10 rows
            except Exception as e:
                return f"SQL Error: {str(e)}"

        database_type = "Azure SQL Database" if use_azure_sql else "SQLite database"

        sql_tool = FunctionTool(
            execute_sql_tool,
            description=f"Execute SQL query on {database_type} with table '{table_name}' and columns: {', '.join(column_names)}",
        )

        system_message = f"""You are a SQL expert that helps write and execute SQL queries on data stored in {database_type}.

{memory_context}IMPORTANT: If there is previous conversation context above, you MUST:
- Reference it when answering follow-up questions
- Use information from previous queries to inform new queries
- Maintain context about what data has already been discussed
- Answer questions that refer to "it", "that", "those" etc. based on previous context

Tasks:
- Write SQL queries to answer user questions about the data
- Use the 'execute_sql_tool' to run queries
- Always consider and reference previous conversation when relevant
- Format your response based on the number of rows:
  - Single Row: Use the format {{column_name: value, column_name: value}}
  - Multiple Rows: Use a list format with each row as a dictionary

The target table '{table_name}' contains the following columns: {", ".join(column_names)}.
Use "SELECT ... FROM {table_name}" format for your queries.
DO NOT change schema or table names.
When composing summary statistics, use functions like AVG(), COUNT(), etc.
When the user asks what columns are available, just list them without running a query.

Example queries:
- SELECT * FROM {table_name} LIMIT 5
- SELECT AVG(salary) FROM {table_name}
- SELECT COUNT(*) FROM {table_name} WHERE department = 'Engineering'
"""

        # Set up the agent team with the SQL assistant and a user proxy
        sql_assistant = AssistantAgent(
            name="sql_assistant",
            system_message=system_message,
            model_client=model_client,
            tools=[sql_tool],
            reflect_on_tool_use=True,
        )

        # Create cancellation token
        cancellation_token = CancellationToken()

        # Prepare user message with context
        user_msg = (
            f"Context: {context}\n\nUser question: {chat_request.user_prompt}"
            if context
            else chat_request.user_prompt
        )

        # Use the SQL assistant directly with on_messages for a simpler interaction
        from autogen_agentchat.messages import TextMessage

        # Send the message directly to the SQL assistant
        response = await sql_assistant.on_messages(
            messages=[TextMessage(content=user_msg, source="user")],
            cancellation_token=cancellation_token,
        )

        # Extract the response content
        final_message = (
            response.chat_message.content
            if response.chat_message
            else "No response generated"
        )

        # Calculate token usage manually since LLMUsageTracker doesn't work with simple flows
        from ingenious.utils.token_counter import num_tokens_from_messages

        try:
            # Estimate tokens from the conversation
            messages_for_counting = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": final_message},
            ]
            total_tokens = num_tokens_from_messages(
                messages_for_counting, model_config.model
            )
            prompt_tokens = num_tokens_from_messages(
                messages_for_counting[:-1], model_config.model
            )
            completion_tokens = total_tokens - prompt_tokens
        except Exception as e:
            logger.warning(f"Token counting failed: {e}")
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0

        # Update memory for future conversations (simplified for local testing)
        # In production, this would use the memory manager

        # Make sure to close the model client connection when done
        await model_client.close()

        # Return the response with proper token counting
        return ChatResponse(
            thread_id=chat_request.thread_id or "",
            message_id=str(uuid.uuid4()),
            agent_response=final_message,
            token_count=total_tokens,
            max_token_count=completion_tokens,
            memory_summary=final_message,
        )
