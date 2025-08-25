from typing import Tuple, cast

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat

from ingenious.client.azure import AzureClientFactory
from ingenious.common.enums import AuthenticationMethod
from ingenious.config import get_config
from ingenious.core.structured_logging import get_logger
from ingenious.models.config import Config

logger = get_logger(__name__)


class ConversationPattern:
    def __init__(
        self,
        default_llm_config: dict[str, object],
        topics: list[str],
        memory_record_switch: bool,
        memory_path: str,
        thread_memory: str,
    ):
        self.default_llm_config = default_llm_config
        self.topics = topics
        self.memory_record_switch = memory_record_switch
        self.memory_path = memory_path
        self.thread_memory = thread_memory
        self.context = ""

        # Create Azure OpenAI model client from config
        self.model_client = (
            AzureClientFactory.create_openai_chat_completion_client_from_params(
                model=str(self.default_llm_config["model"]),
                base_url=str(self.default_llm_config["base_url"]),
                api_version=str(self.default_llm_config["api_version"]),
                deployment=str(self.default_llm_config.get("deployment")),
                authentication_method=AuthenticationMethod(
                    self.default_llm_config.get(
                        "authentication_method", AuthenticationMethod.DEFAULT_CREDENTIAL
                    )
                ),
                api_key=str(self.default_llm_config.get("api_key", "")),
            )
        )

        # Initialize memory manager for cloud storage support
        from ingenious.services.memory_manager import (
            get_memory_manager,
            run_async_memory_operation,
        )

        self.memory_manager = get_memory_manager(
            cast(Config, get_config()), memory_path
        )

        # Initialize context file
        if not self.thread_memory:
            run_async_memory_operation(
                self.memory_manager.write_memory(
                    "New conversation. Continue based on user question."
                )
            )

        if self.memory_record_switch and self.thread_memory:
            logger.debug(
                "Memory recording enabled",
                thread_memory_length=len(self.thread_memory),
                note="Requires ChatHistorySummariser for optional dependency",
            )
            run_async_memory_operation(
                self.memory_manager.write_memory(self.thread_memory)
            )

        # Read current context
        self.context = run_async_memory_operation(
            self.memory_manager.read_memory(default_content="")
        )

        # Simplified: Just one classifier agent that does both classification and response
        self.classifier = AssistantAgent(
            name="classifier",
            model_client=self.model_client,
            system_message=(
                "You are a classification and response agent. Classify user messages into these categories:\n"
                "1. payload_type_1: General product inquiries, features, specifications\n"
                "2. payload_type_2: Purchase-related questions, pricing, availability\n"
                "3. payload_type_3: Support issues, problems, complaints\n"
                "4. undefined: Messages that don't fit the above categories\n\n"
                "Respond with a JSON object containing:\n"
                "- 'category': the classification category\n"
                "- 'explanation': brief reason for classification\n"
                "- 'response': helpful response to the user\n"
                "Then say TERMINATE to end the conversation."
            ),
        )

        # Simple user proxy to start the conversation
        self.user_proxy = UserProxyAgent(name="user_proxy")

    def add_topic_agent(self, agent_name: str, system_message: str) -> None:
        """Add a topic agent - simplified to do nothing since we use single classifier"""
        pass

    async def get_conversation_response(self, input_message: str) -> Tuple[str, str]:
        """
        Simplified conversation with just classifier + user proxy in round-robin (max 2 turns)
        """
        try:
            # Create a simple round-robin team with just 2 agents
            team = RoundRobinGroupChat(participants=[self.user_proxy, self.classifier])

            # Run with a much simpler task
            result = await team.run(
                task=f"Classify and respond to this message: {input_message}"
            )

            # Get the final message content
            final_message = "No response"
            if result.messages:
                last_msg = result.messages[-1]
                if hasattr(last_msg, "content"):
                    final_message = str(last_msg.content)
                else:
                    final_message = str(last_msg)

            # Update context using MemoryManager
            from ingenious.services.memory_manager import run_async_memory_operation

            run_async_memory_operation(self.memory_manager.write_memory(final_message))
            self.context = final_message

            return final_message, self.context

        except Exception as e:
            logger.error("Error in conversation response", error=str(e), exc_info=True)
            error_response = {
                "category": "payload_type_1",
                "explanation": "Error occurred during classification",
                "response": "I apologize, but I encountered an issue. How can I help you today?",
            }
            return str(error_response), str(e)

    async def close(self) -> None:
        """Close the model client connection"""
        await self.model_client.close()
