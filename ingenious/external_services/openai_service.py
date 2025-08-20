import re
from typing import AsyncIterator

from openai import NOT_GIVEN, BadRequestError
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)

from ingenious.client.azure import AzureClientFactory
from ingenious.common.enums import AuthenticationMethod
from ingenious.core.structured_logging import get_logger
from ingenious.errors.content_filter_error import ContentFilterError
from ingenious.errors.token_limit_exceeded_error import TokenLimitExceededError

logger = get_logger(__name__)


class OpenAIService:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str,
        open_ai_model: str,
        deployment: str = "",
        authentication_method: AuthenticationMethod = AuthenticationMethod.DEFAULT_CREDENTIAL,
        client_id: str = "",
        client_secret: str = "",
        tenant_id: str = "",
    ):
        # Use the centralized Azure client factory for consistent validation and authentication
        self.client = AzureClientFactory.create_openai_client_from_params(
            model=open_ai_model,
            base_url=azure_endpoint,
            api_version=api_version,
            deployment=deployment,
            api_key=api_key,
            authentication_method=authentication_method,
            client_id=client_id,
            client_secret=client_secret,
            tenant_id=tenant_id,
        )

        self.model = open_ai_model

    async def generate_response(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        json_mode=False,
    ) -> ChatCompletionMessage:
        logger.debug(
            "Generating OpenAI response",
            model=self.model,
            message_count=len(messages),
            has_tools=tools is not None,
            json_mode=json_mode,
        )
        try:
            # Handle tool_choice parameter properly
            effective_tool_choice = (
                tool_choice
                if tool_choice is not None
                else ("auto" if tools else NOT_GIVEN)
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or NOT_GIVEN,
                tool_choice=effective_tool_choice,
                response_format={"type": "json_object"} if json_mode else NOT_GIVEN,
                temperature=0.2,
            )
            return response.choices[0].message
        except BadRequestError as error:
            # Log the error with structured context
            logger.error(
                "OpenAI API request failed",
                error_type="BadRequestError",
                error_code=error.code,
                error_message=error.message,
                model=self.model,
                exc_info=True,
            )

            # Default to the general message from the exception
            message = error.message

            # Check if the body is a dictionary and refine the message if possible
            if isinstance(error.body, dict):
                message = error.body.get("message", message)

                # Check for content filter specific errors
                if error.code == "content_filter" and "innererror" in error.body:
                    content_filter_results = error.body["innererror"].get(
                        "content_filter_result", {}
                    )
                    raise ContentFilterError(message, content_filter_results)

                # Check for token limit errors
                token_error_pattern = (
                    r"This model's maximum context length is (\d+) tokens, "
                    r"however you requested (\d+) tokens \((\d+) in your prompt; "
                    r"(\d+) for the completion\). Please reduce your prompt; or "
                    r"completion length."
                )
                token_error_match = re.match(token_error_pattern, message)
                if token_error_match:
                    (
                        max_context_length,
                        requested_tokens,
                        prompt_tokens,
                        completion_tokens,
                    ) = token_error_match.groups()
                    raise TokenLimitExceededError(
                        message=message,
                        max_context_length=int(max_context_length),
                        requested_tokens=int(requested_tokens),
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                    )

            raise Exception(message)
        except Exception as e:
            logger.exception(e)
            raise

    async def generate_streaming_response(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        json_mode=False,
    ) -> AsyncIterator[str]:
        """Generate streaming response from OpenAI API.

        Yields content chunks as they are received from the OpenAI API.
        """
        logger.debug(
            "Generating streaming OpenAI response",
            model=self.model,
            message_count=len(messages),
            has_tools=tools is not None,
            json_mode=json_mode,
        )
        try:
            # Handle tool_choice parameter properly
            effective_tool_choice = (
                tool_choice
                if tool_choice is not None
                else ("auto" if tools else NOT_GIVEN)
            )

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or NOT_GIVEN,
                tool_choice=effective_tool_choice,
                response_format={"type": "json_object"} if json_mode else NOT_GIVEN,
                temperature=0.2,
                stream=True,
            )

            for chunk in response:
                if (
                    chunk.choices
                    and chunk.choices[0].delta
                    and chunk.choices[0].delta.content
                ):
                    yield chunk.choices[0].delta.content

        except BadRequestError as error:
            # Same error handling as non-streaming version
            logger.error(
                "OpenAI streaming API request failed",
                error_type="BadRequestError",
                error_code=error.code,
                error_message=error.message,
                model=self.model,
                exc_info=True,
            )

            message = error.message
            if isinstance(error.body, dict):
                message = error.body.get("message", message)

                if error.code == "content_filter" and "innererror" in error.body:
                    content_filter_results = error.body["innererror"].get(
                        "content_filter_result", {}
                    )
                    raise ContentFilterError(message, content_filter_results)

                token_error_pattern = (
                    r"This model's maximum context length is (\d+) tokens, "
                    r"however you requested (\d+) tokens \((\d+) in your prompt; "
                    r"(\d+) for the completion\). Please reduce your prompt; or "
                    r"completion length."
                )
                token_error_match = re.match(token_error_pattern, message)
                if token_error_match:
                    (
                        max_context_length,
                        requested_tokens,
                        prompt_tokens,
                        completion_tokens,
                    ) = token_error_match.groups()
                    raise TokenLimitExceededError(
                        message=message,
                        max_context_length=int(max_context_length),
                        requested_tokens=int(requested_tokens),
                        prompt_tokens=int(prompt_tokens),
                        completion_tokens=int(completion_tokens),
                    )

            raise Exception(message)
        except Exception as e:
            logger.exception(e)
            raise
