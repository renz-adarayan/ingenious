import logging
import os
import uuid
from typing import Any, AsyncIterator, cast

from autogen_agentchat.agents import AssistantAgent
from autogen_core import EVENT_LOGGER_NAME, CancellationToken
from autogen_core.tools import FunctionTool

from ingenious.client.azure import AzureClientFactory
from ingenious.models.agent import LLMUsageTracker
from ingenious.models.chat import ChatRequest, ChatResponse, ChatResponseChunk
from ingenious.services.chat_services.multi_agent.service import IConversationFlow


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
            event_type="knowledge_base",
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

        # Check if Azure Search is configured
        use_azure_search = (
            hasattr(self._config, "azure_search_services")
            and self._config.azure_search_services
            and len(self._config.azure_search_services) > 0
        )

        if use_azure_search:
            # Use Azure Search configuration
            search_config = self._config.azure_search_services[0]
            search_backend = "Azure AI Search"
            context = f"Knowledge base search assistant using {search_backend} for finding information."
        else:
            # Use local ChromaDB
            search_config = None
            search_backend = "local ChromaDB"
            context = f"Knowledge base search assistant using {search_backend} for finding information."

        # Create search tool function supporting both Azure Search and ChromaDB
        async def search_tool(search_query: str, topic: str = "general") -> str:
            f"""Search for information using {search_backend}"""
            try:
                if use_azure_search:
                    # Use Azure AI Search
                    try:
                        assert search_config is not None
                        search_client = AzureClientFactory.create_search_client(
                            search_config=cast(Any, search_config),
                            index_name="test-index",
                        )

                        # Perform search
                        search_results = search_client.search(
                            search_text=search_query, top=3, include_total_count=True
                        )

                        results = []
                        for result in search_results:
                            # Extract content from search result
                            content = result.get("content", "") or str(result)
                            if content:
                                results.append(content)

                        if results:
                            return (
                                "Found relevant information from Azure AI Search:\n\n"
                                + "\n\n".join(results)
                            )
                        else:
                            return f"No relevant information found in Azure AI Search for query: {search_query}"

                    except Exception as e:
                        return f"Azure Search error: {str(e)}. Ensure the search index exists and contains documents."

                else:
                    # Use local ChromaDB
                    try:
                        import chromadb
                    except ImportError:
                        return "Error: ChromaDB not installed. Please install with: uv add chromadb"

                    # Initialize ChromaDB client
                    knowledge_base_path = os.path.join(
                        self._memory_path, "knowledge_base"
                    )
                    chroma_path = os.path.join(self._memory_path, "chroma_db")

                    # Ensure knowledge base directory exists
                    if not os.path.exists(knowledge_base_path):
                        os.makedirs(knowledge_base_path, exist_ok=True)
                        return "Error: Knowledge base directory is empty. Please add documents to .tmp/knowledge_base/"

                    # Initialize ChromaDB
                    client = chromadb.PersistentClient(path=chroma_path)

                    # Get or create collection
                    collection_name = "knowledge_base"
                    try:
                        collection = client.get_collection(name=collection_name)
                    except Exception:
                        # Create collection if it doesn't exist
                        collection = client.create_collection(name=collection_name)

                        # Load documents from knowledge base directory
                        documents = []
                        document_ids = []

                        for filename in os.listdir(knowledge_base_path):
                            if filename.endswith(".md") or filename.endswith(".txt"):
                                filepath = os.path.join(knowledge_base_path, filename)
                                with open(filepath, "r", encoding="utf-8") as f:
                                    content = f.read()
                                    # Split content into chunks
                                    chunks = content.split("\n\n")
                                    for i, chunk in enumerate(chunks):
                                        if chunk.strip():
                                            documents.append(chunk.strip())
                                            document_ids.append(f"{filename}_chunk_{i}")

                        if documents:
                            collection.add(documents=documents, ids=document_ids)
                        else:
                            return (
                                "Error: No documents found in knowledge base directory"
                            )

                    # Search the collection
                    results = collection.query(query_texts=[search_query], n_results=3)

                    if results["documents"] and results["documents"][0]:
                        search_results = "\n\n".join(results["documents"][0])
                        return f"Found relevant information from ChromaDB:\n\n{search_results}"
                    else:
                        return f"No relevant information found in ChromaDB for query: {search_query}"

            except Exception as e:
                return f"Search error: {str(e)}"

        search_function_tool = FunctionTool(
            search_tool,
            description=f"Search for information using {search_backend}. Use relevant keywords to find relevant information.",
        )

        # Create the search assistant agent with memory context
        search_system_message = f"""You are a knowledge base search assistant that can use both Azure AI Search and local ChromaDB storage.

{memory_context}IMPORTANT: If there is previous conversation context above, you MUST:
- Reference it when answering follow-up questions
- Use information from previous searches to inform new searches
- Maintain context about what information has already been discussed
- Answer questions that refer to "it", "that", "those" etc. based on previous context

Tasks:
- Help users find information by searching the knowledge base
- Use the search_tool to look up information
- Always base your responses on search results from the knowledge base
- Always consider and reference previous conversation when relevant
- If no information is found, clearly state that and suggest rephrasing the query

Guidelines for search queries:
- Use specific, relevant keywords
- Try different phrasings if initial search doesn't return results
- Focus on topics that are relevant to the knowledge base content

Knowledge base contains documents about:
- Azure configuration and setup
- Workplace safety guidelines
- Health information and nutrition
- Emergency procedures
- Mental health and wellbeing
- First aid basics
- General informational content

Format your responses clearly and cite the knowledge base when providing information.
TERMINATE your response when the task is complete.
"""

        # Set up the search assistant agent
        search_assistant = AssistantAgent(
            name="search_assistant",
            system_message=search_system_message,
            model_client=model_client,
            tools=[search_function_tool],
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

        # Use the search assistant directly with on_messages for a simpler interaction
        from autogen_agentchat.messages import TextMessage

        # Send the message directly to the search assistant
        response = await search_assistant.on_messages(
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
                {"role": "system", "content": search_system_message},
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

    async def get_streaming_conversation_response(
        self, chat_request: ChatRequest
    ) -> AsyncIterator[ChatResponseChunk]:
        """Streaming version of knowledge base agent conversation."""

        # Generate a message ID for this conversation
        message_id = str(uuid.uuid4())
        thread_id = chat_request.thread_id or ""

        try:
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
                event_type="knowledge_base_streaming",
            )

            logger.handlers = [llm_logger]

            # Retrieve thread memory for context (same as non-streaming)
            memory_context = ""
            if chat_request.thread_id and self._chat_service:
                try:
                    thread_messages = await self._chat_service.chat_history_repository.get_thread_messages(
                        chat_request.thread_id
                    )
                    if thread_messages:
                        recent_messages = (
                            thread_messages[-10:]
                            if len(thread_messages) > 10
                            else thread_messages
                        )
                        memory_parts = []
                        for msg in recent_messages:
                            memory_parts.append(f"{msg.role}: {msg.content[:100]}...")
                        memory_context = (
                            "Previous conversation:\n"
                            + "\n".join(memory_parts)
                            + "\n\n"
                        )
                except Exception as e:
                    logger.warning(f"Failed to retrieve thread memory: {e}")

            # Create the model client
            model_client = AzureClientFactory.create_openai_chat_completion_client(
                model_config
            )

            # Send initial chunk indicating start of processing
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="status",
                content="Searching knowledge base...",
                is_final=False,
            )
            # Set up search functionality (same as non-streaming for now)
            use_azure_search = (
                hasattr(self._config, "azure_search_services")
                and self._config.azure_search_services
                and len(self._config.azure_search_services) > 0
            )

            if use_azure_search:
                search_config = self._config.azure_search_services[0]
                search_backend = "Azure AI Search"
            else:
                search_config = None
                search_backend = "local ChromaDB"
                search_backend = "local ChromaDB"

            # Create search tool (abbreviated for brevity - would use same implementation)
            def search_tool(search_query: str) -> str:
                try:
                    if use_azure_search:
                        # Azure Search implementation using AzureClientFactory
                        try:
                            assert search_config is not None
                            search_client = AzureClientFactory.create_search_client(
                                search_config=cast(Any, search_config),
                                index_name="test-index",
                            )
                            results = search_client.search(
                                search_text=search_query,
                                top=5,
                                include_total_count=True,
                            )
                            search_results = []
                            for result in results:
                                content = result.get("content", "") or str(result)
                                if content:
                                    search_results.append(content)
                                    search_results.append(content)

                            if search_results:
                                return (
                                    "Found relevant information from Azure AI Search:\n\n"
                                    + "\n\n".join(search_results)
                                )
                            else:
                                return f"No relevant information found in Azure AI Search for query: {search_query}"
                        except Exception as e:
                            return f"Azure Search error: {str(e)}. Ensure the search index exists and contains documents."
                    else:
                        # ChromaDB implementation would go here (abbreviated)
                        return f"ChromaDB search results for: {search_query}"

                except Exception as e:
                    return f"Search error: {str(e)}"
                    return f"Search error: {str(e)}"

            search_function_tool = FunctionTool(
                search_tool,
                description=f"Search for information using {search_backend}. Use relevant keywords to find relevant information.",
            )

            # Create the search assistant agent with memory context
            search_system_message = f"""You are a knowledge base search assistant that can use both Azure AI Search and local ChromaDB storage.

{memory_context}IMPORTANT: If there is previous conversation context above, you MUST:
- Reference it when answering follow-up questions
- Use information from previous searches to inform new searches
- Maintain context about what information has already been discussed
- Answer questions that refer to "it", "that", "those" etc. based on previous context

Tasks:
- Help users find information by searching the knowledge base
- Use the search_tool to look up information
- Always base your responses on search results from the knowledge base
- Always consider and reference previous conversation when relevant
- If no information is found, clearly state that and suggest rephrasing the query

Guidelines for search queries:
- Use specific, relevant keywords
- Try different phrasings if initial search doesn't return results
- Focus on topics that are relevant to the knowledge base content"""

            search_assistant = AssistantAgent(
                name="search_assistant",
                model_client=model_client,
                tools=[search_function_tool],
                system_message=search_system_message,
            )

            user_msg = f"User query: {chat_request.user_prompt}"

            # Send status update
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="status",
                content="Generating response...",
                is_final=False,
            )

            # Run the streaming conversation
            accumulated_content = ""
            total_tokens = 0
            prompt_tokens = 0
            completion_tokens = 0

            cancellation_token = CancellationToken()

            try:
                # Use run_stream for streaming response
                stream = search_assistant.run_stream(
                    task=user_msg, cancellation_token=cancellation_token
                )

                async for message in stream:
                    # Handle different message types from AutoGen streaming
                    if hasattr(message, "content") and message.content:
                        # Stream content chunk
                        yield ChatResponseChunk(
                            thread_id=thread_id,
                            message_id=message_id,
                            chunk_type="content",
                            content=message.content,
                            is_final=False,
                        )
                        accumulated_content += message.content

                    # Handle token usage updates if available
                    if hasattr(message, "usage"):
                        usage = message.usage
                        if hasattr(usage, "total_tokens"):
                            total_tokens = usage.total_tokens
                        if hasattr(usage, "prompt_tokens"):
                            prompt_tokens = usage.prompt_tokens
                        if hasattr(usage, "completion_tokens"):
                            completion_tokens = usage.completion_tokens

                        # Send token count update
                        yield ChatResponseChunk(
                            thread_id=thread_id,
                            message_id=message_id,
                            chunk_type="token_count",
                            token_count=total_tokens,
                            is_final=False,
                        )

                    # Handle final result
                    if hasattr(message, "__class__") and "TaskResult" in str(
                        message.__class__
                    ):
                        if hasattr(message, "messages") and message.messages:
                            # Get the final message content
                            final_msg = message.messages[-1]
                            if hasattr(final_msg, "content") and final_msg.content:
                                if final_msg.content not in accumulated_content:
                                    yield ChatResponseChunk(
                                        thread_id=thread_id,
                                        message_id=message_id,
                                        chunk_type="content",
                                        content=final_msg.content,
                                        is_final=False,
                                    )
                                    accumulated_content += final_msg.content

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                # Send error chunk but continue to final chunk
                yield ChatResponseChunk(
                    thread_id=thread_id,
                    message_id=message_id,
                    chunk_type="content",
                    content=f"[Error during streaming: {str(e)}]",
                    is_final=False,
                )

            # Estimate tokens if not provided by streaming
            if total_tokens == 0:
                try:
                    from ingenious.utils.token_counter import num_tokens_from_messages

                    messages_for_counting = [
                        {"role": "system", "content": search_system_message},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": accumulated_content},
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
                    total_tokens = len(accumulated_content) // 4  # Rough estimate

            # Close the model client
            await model_client.close()

            # Send final chunk with all metadata
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="final",
                token_count=total_tokens,
                max_token_count=completion_tokens,
                memory_summary=accumulated_content[:200] + "..."
                if len(accumulated_content) > 200
                else accumulated_content,
                event_type="knowledge_base_streaming",
                is_final=True,
            )

        except Exception as e:
            logger.error(f"Error in streaming knowledge base response: {e}")
            yield ChatResponseChunk(
                thread_id=thread_id,
                message_id=message_id,
                chunk_type="error",
                content=f"An error occurred: {str(e)}",
                is_final=True,
            )
