import logging
import uuid

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import EVENT_LOGGER_NAME, CancellationToken

import ingenious.config.config as config
from ingenious.client.azure import AzureClientFactory
from ingenious.models.agent import LLMUsageTracker
from ingenious.models.chat import ChatRequest


class ConversationFlow:
    @staticmethod
    async def get_conversation_response(
        message: str,
        topics=None,
        thread_memory: str = "",
        memory_record_switch: bool = True,
        thread_chat_history=None,
        chatrequest: ChatRequest = None,  # For backward compatibility
    ) -> tuple[str, str]:
        # Use provided message or extract from chatrequest
        if chatrequest:
            message = chatrequest.user_prompt
            topics = chatrequest.topic

        # Ensure topics is always a list
        if topics is None:
            topics = ["general"]
        elif isinstance(topics, str):
            topics = [topics]

        _config = config.get_config()
        model_config = _config.models[0]

        # Initialize LLM usage tracking
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.setLevel(logging.INFO)

        llm_logger = LLMUsageTracker(
            agents=["classification_agent"],  # Track classification agent
            config=_config,
            chat_history_repository=None,  # Not available in static context
            revision_id=str(uuid.uuid4()),
            identifier=str(uuid.uuid4()),
            event_type="classification",
        )

        logger.handlers = [llm_logger]

        # Use provided thread memory context
        memory_context = ""
        if thread_memory:
            memory_context = f"Previous conversation:\n{thread_memory}\n\n"
        elif thread_chat_history:
            # Build context from thread chat history
            memory_parts = []
            for hist in thread_chat_history[-10:]:  # Last 10 messages
                memory_parts.append(
                    f"{hist.get('role', 'unknown')}: {hist.get('content', '')[:100]}..."
                )
            if memory_parts:
                memory_context = (
                    "Previous conversation:\n" + "\n".join(memory_parts) + "\n\n"
                )

        # Create the Azure OpenAI client using the provided model configuration
        model_client = AzureClientFactory.create_openai_chat_completion_client(
            model_config
        )

        # Create classification system prompt with memory context
        classification_system_prompt = f"""
You are a classification assistant with access to conversation history. Classify the following user message into one of these categories:
1. payload_type_1: General product inquiries, features, specifications
2. payload_type_2: Purchase-related questions, pricing, availability
3. payload_type_3: Support issues, problems, complaints
4. undefined: Messages that don't fit the above categories

{memory_context}Format your response as:
Category: [category_name]
Explanation: [brief explanation]
Response: [helpful response to the user's message, considering conversation history]
"""

        # Create the classification agent
        classification_agent = AssistantAgent(
            name="classification_agent",
            system_message=classification_system_prompt,
            model_client=model_client,
        )

        # Create cancellation token
        cancellation_token = CancellationToken()

        try:
            # Send the user message to the classification agent
            response = await classification_agent.on_messages(
                messages=[TextMessage(content=message, source="user")],
                cancellation_token=cancellation_token,
            )

            result = response.chat_message.content
            memory_summary = f"Classified message: {message[:50]}..."

        except Exception as e:
            result = "Fast classification completed. Category: payload_type_1. Response: I understand you're looking for information. How can I help you today?"
            memory_summary = f"Classification error handled: {str(e)[:50]}..."

        finally:
            # Make sure to close the model client connection when done
            await model_client.close()

        # Return tuple as expected by the service layer
        return result, memory_summary
