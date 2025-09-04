import asyncio
from abc import ABC
from typing import List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import TextMessage
from autogen_core import MessageContext, RoutedAgent, TopicId, message_handler
from autogen_core.models import (
    AssistantMessage,
    FunctionExecutionResultMessage,
    LLMMessage,
    ModelFamily,
    ModelInfo,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from ingenious.models.agent import (
    Agent,
    AgentChat,
    AgentMessage,
)

# Custom model info for gpt-5-mini
GPT_5_MINI_MODEL_INFO: ModelInfo = {
    "vision": True,
    "function_calling": True,
    "json_output": True,
    "family": ModelFamily.GPT_4O,  # Use GPT_4O family as it's similar
    "structured_output": True,
    "multiple_system_messages": True,
}


class RoutedAssistantAgent(RoutedAgent, ABC):
    def __init__(
        self, agent: Agent, data_identifier: str, next_agent_topic: str = None, tools=[]
    ) -> None:
        super().__init__(agent.agent_name)

        # Map model config parameters to AzureOpenAIChatCompletionClient parameters
        azure_config = {
            "model": agent.model.model,
            "api_key": agent.model.api_key,
            "azure_endpoint": agent.model.base_url,
            "azure_deployment": agent.model.deployment or agent.model.model,
            "api_version": agent.model.api_version,
        }

        # Add custom model_info for gpt-5-mini
        if agent.model.model == "gpt-5-mini":
            azure_config["model_info"] = GPT_5_MINI_MODEL_INFO

        self._model_client = AzureOpenAIChatCompletionClient(**azure_config)
        assistant_agent = AssistantAgent(
            name=agent.agent_name,
            system_message=agent.system_prompt,
            description="I am an AI assistant that helps with research.",
            model_client=self._model_client,
        )
        self._delegate = assistant_agent
        self._agent: Agent = agent
        self._data_identifier = data_identifier
        self._next_agent_topic = next_agent_topic
        self._tools = tools
        self._system_messages = [SystemMessage(content=agent.system_prompt)]

    @message_handler
    async def handle_my_message_type(
        self, message: AgentMessage, ctx: MessageContext
    ) -> None:
        """
        Receives the message and sends it to the assistant agent to make a llm call. It then calls publish message to send the response to the next agent.
        """
        print(self._agent.agent_name)
        agent_chat = self._agent.add_agent_chat(
            content=message.content, identifier=self._data_identifier, ctx=ctx
        )
        # agent_chat.chat_response = await self._delegate.on_messages(
        #     messages=[TextMessage(content=message.content, source=ctx.topic_id.source)],
        #     cancellation_token=ctx.cancellation_token
        # )

        # if self._next_agent_topic:
        #     await self.publish_my_message(agent_chat)

        # Create a session of messages.
        session: List[LLMMessage] = self._system_messages + [
            UserMessage(content=message.content, source="user")
        ]
        execute_tool_calls = True

        # Run the chat completion with the tools.
        create_result = await self._model_client.create(
            messages=session,
            tools=self._tools,
            cancellation_token=ctx.cancellation_token,
        )

        # If there are no tool calls, return the result.
        if isinstance(create_result.content, str):
            agent_chat.chat_response = Response(
                chat_message=TextMessage(content=create_result.content, source="user")
            )
            execute_tool_calls = False

        if execute_tool_calls:
            # Add the first model create result to the session.
            session.append(
                AssistantMessage(content=create_result.content, source="assistant")
            )

            # Execute the tool calls.
            results = await asyncio.gather(
                *[
                    self._agent.execute_tool_call(
                        call, ctx.cancellation_token, tools=self._tools
                    )
                    for call in create_result.content
                ]
            )

            # Add the function execution results to the session.
            session.append(FunctionExecutionResultMessage(content=results))

            # Run the chat completion again to reflect on the history and function execution results.
            create_result = await self._model_client.create(
                messages=session,
                cancellation_token=ctx.cancellation_token,
            )
            assert isinstance(create_result.content, str)

            # Return the result as a message.
            agent_chat.chat_response = Response(
                chat_message=TextMessage(content=create_result.content, source="user")
            )

        if self._next_agent_topic:
            await self.publish_my_message(agent_chat)

    async def publish_my_message(self, agent_chat: AgentChat) -> None:
        """
        Publishes the response to the next agent.
        """
        # Publish the outgoing message to the next agent
        await self.publish_message(
            AgentMessage(content=agent_chat.chat_response.chat_message.content),
            topic_id=TopicId(type=self._next_agent_topic, source=self.id.key),
        )
