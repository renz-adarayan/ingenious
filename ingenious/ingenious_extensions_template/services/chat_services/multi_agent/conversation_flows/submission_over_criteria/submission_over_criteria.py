import asyncio
import json
import logging
from typing import Annotated, List

import jsonpickle
from autogen_core import (
    EVENT_LOGGER_NAME,
    SingleThreadedAgentRuntime,
    TopicId,
    TypeSubscription,
)
from autogen_core.tools import FunctionTool
from ingenious.models.ag_agents import (
    RelayAgent,
)
from ingenious.models.agent import (
    AgentMessage,
    LLMUsageTracker,
)
from ingenious.models.chat import ChatRequest, ChatResponse
from ingenious.models.message import Message as ChatHistoryMessage
from ingenious.services.chat_services.multi_agent.service import IConversationFlow

from ingenious_extensions_template.models.submission_over_criteria.ag_agents import (
    RoutedAssistantAgent,
)

# Custom class import from ingenious_extensions
from ingenious_extensions_template.models.submission_over_criteria.agent import ProjectAgents
from ingenious_extensions_template.models.submission_over_criteria.submissions import (
    SubmissionOverCriteriaRequest,
)


class ConversationFlow(IConversationFlow):
    async def get_conversation_response(
        self,
        chat_request: ChatRequest,
    ) -> ChatResponse:
        try:
            message = json.loads(chat_request.user_prompt)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"submission_over_criteria workflow requires JSON-formatted data. "
                f"Please provide a valid JSON string with fields: revision_id, identifier, submissions, criteria. "
                f'Example: {{"revision_id": "test-v1", "identifier": "eval-001", "submissions": [...], "criteria": [...]}}\\n'
                f"JSON parsing error: {str(e)}"
            ) from e

        # Validate required fields
        required_fields = ["revision_id", "identifier", "submissions", "criteria"]
        missing_fields = [field for field in required_fields if field not in message]
        if missing_fields:
            raise ValueError(
                f"submission_over_criteria workflow requires the following fields in JSON data: {', '.join(missing_fields)}. "
                f"Current data contains: {list(message.keys())}. "
                f"Please include all required fields: revision_id, identifier, submissions, criteria"
            )

        # Get your agents and agent chats from your custom class in models folder
        project_agents = ProjectAgents()
        agents = project_agents.Get_Project_Agents(self._config)

        # Process your data payload using your custom data model class
        try:
            submission_request = SubmissionOverCriteriaRequest.model_validate(message)
        except Exception as e:
            raise ValueError(
                f"submission_over_criteria workflow data validation failed. "
                f"Please ensure your JSON data matches the expected schema for submission evaluation. "
                f"Validation error: {str(e)}"
            ) from e

        # Get the revision id and identifier from the message payload
        revision_id = message["revision_id"]
        identifier = message["identifier"]

        # Instantiate the logger and handler
        logger = logging.getLogger(EVENT_LOGGER_NAME)
        logger.setLevel(logging.INFO)

        llm_logger = LLMUsageTracker(
            agents=agents,
            config=self._config,
            chat_history_repository=self._chat_service.chat_history_repository,
            revision_id=revision_id,
            identifier=identifier,
            event_type="default",
        )

        logger.handlers = [llm_logger]

        self._logger.debug("Starting Submission Over Criteria Flow")

        # Now add your system prompts to your agents from the prompt templates
        for agent in agents.get_agents():
            template_name = f"{agent.agent_name}_prompt.jinja"
            agent.system_prompt = await self.Get_Template(
                file_name=template_name, revision_id=revision_id
            )

        # Now construct your autogen conversation pattern
        runtime = SingleThreadedAgentRuntime()

        # Tool for calculating weighted scores
        async def calculate_weighted_score(
            scores: Annotated[List[float], "List of individual scores"],
            weights: Annotated[List[float], "List of weights for each score"],
        ) -> float:
            """Calculate weighted average score"""
            if len(scores) != len(weights):
                return 0.0
            total_weight = sum(weights)
            if total_weight == 0:
                return 0.0
            weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
            return weighted_sum / total_weight

        scoring_tool = FunctionTool(
            calculate_weighted_score,
            description="Calculate weighted average score from individual scores and weights.",
        )

        async def register_research_agent(
            agent_name: str,
            tools: List[FunctionTool] = [],
            next_agent_topic: str = None,
        ):
            agent = agents.get_agent_by_name(agent_name=agent_name)
            reg_agent = await RoutedAssistantAgent.register(
                runtime=runtime,
                type=agent.agent_name,
                factory=lambda: RoutedAssistantAgent(
                    agent=agent,
                    data_identifier=identifier,
                    next_agent_topic=next_agent_topic,
                    tools=tools,
                ),
            )
            await runtime.add_subscription(
                TypeSubscription(topic_type=agent_name, agent_type=reg_agent.type)
            )

        # Register evaluation agents in proper sequence
        await register_research_agent(
            agent_name="criteria_analyzer_agent", next_agent_topic="user_proxy"
        )
        await register_research_agent(
            agent_name="submission_evaluator_agent", next_agent_topic="user_proxy"
        )
        await register_research_agent(
            agent_name="scoring_agent",
            tools=[scoring_tool],
            next_agent_topic="user_proxy",
        )
        await register_research_agent(agent_name="ranking_agent", next_agent_topic=None)

        # User proxy to coordinate the workflow
        user_proxy = await RelayAgent.register(
            runtime,
            "user_proxy",
            lambda: RelayAgent(
                agents.get_agent_by_name("user_proxy"),
                data_identifier=identifier,
                next_agent_topic="ranking_agent",  # Pass to ranking agent
                number_of_messages_before_next_agent=3,  # Wait for 3 analyzer outputs
            ),
        )
        await runtime.add_subscription(
            TypeSubscription(topic_type="user_proxy", agent_type=user_proxy.type)
        )

        # Skip chat history injection for submission evaluations to ensure independence
        # Each evaluation should be performed without contamination from previous results

        runtime.start()

        # Prepare messages for different agents
        initial_message: AgentMessage = AgentMessage(content=json.dumps(message))
        initial_message.content = "```json\\n" + initial_message.content + "\\n```"

        criteria_message: AgentMessage = AgentMessage(
            content=submission_request.display_criteria_as_table()
        )

        submissions_message: AgentMessage = AgentMessage(
            content=submission_request.display_submissions_as_table()
        )

        # Start the evaluation workflow by sending messages to initial agents
        # Ranking agent will receive data from scoring agent automatically
        await asyncio.gather(
            runtime.publish_message(
                criteria_message,
                topic_id=TopicId(type="criteria_analyzer_agent", source="default"),
            ),
            runtime.publish_message(
                submissions_message,
                topic_id=TopicId(type="submission_evaluator_agent", source="default"),
            ),
            runtime.publish_message(
                initial_message,
                topic_id=TopicId(type="scoring_agent", source="default"),
            ),
        )

        await runtime.stop_when_idle()

        # Write LLM responses to file for prompt tuning
        await llm_logger.write_llm_responses_to_file(
            file_prefixes=[str(chat_request.user_id)]
        )

        # Lastly return your chat response object
        chat_response = ChatResponse(
            thread_id=chat_request.thread_id,
            message_id=identifier,
            agent_response=jsonpickle.encode(
                unpicklable=False, value=llm_logger._queue
            ),
            token_count=llm_logger.prompt_tokens,
            max_token_count=0,
            memory_summary="",
        )

        # Get the ranking response for storage (instead of summary)
        ranking_response = None
        for chat in llm_logger._queue:
            if chat.chat_name == "ranking_agent":
                ranking_response = chat
                break

        if ranking_response:
            message: ChatHistoryMessage = ChatHistoryMessage(
                user_id=chat_request.user_id,
                thread_id=chat_request.thread_id,
                message_id=identifier,
                role="output",
                content=ranking_response.chat_response.chat_message.content,
                content_filter_results=None,
                tool_calls=None,
                tool_call_id=None,
                tool_call_function=None,
            )

            _ = await self._chat_service.chat_history_repository.add_message(
                message=message
            )
        else:
            # If no ranking found, create a default message
            message: ChatHistoryMessage = ChatHistoryMessage(
                user_id=chat_request.user_id,
                thread_id=chat_request.thread_id,
                message_id=identifier,
                role="output",
                content="Submission evaluation completed. See agent_response for detailed results.",
                content_filter_results=None,
                tool_calls=None,
                tool_call_id=None,
                tool_call_function=None,
            )

            _ = await self._chat_service.chat_history_repository.add_message(
                message=message
            )

        return chat_response
