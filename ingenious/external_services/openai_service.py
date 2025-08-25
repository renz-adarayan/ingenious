# ingenious/external_services/openai_service.py
from __future__ import annotations

import re
from typing import Any, AsyncIterator, Optional

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
        *,
        client: Optional[Any] = None,
    ):
        """
        Initialize the service.

        Notes:
        - For Azure OpenAI, `deployment` is the identifier you pass as `model=...`
          to the Chat Completions API. If not provided, we default it to `open_ai_model`.
        - You may inject a prebuilt `client` (useful in tests).
        """
        # Keep both for clarity: base model (for logs) and deployment (for API).
        self.model = open_ai_model
        self._deployment = deployment or open_ai_model

        if client is not None:
            self.client = client  # dependency injection for tests/advanced use
            return

        # Centralized client creation (lazy imports, shared auth rules)
        self.client = AzureClientFactory.create_openai_client_from_params(
            model=open_ai_model,
            base_url=azure_endpoint,
            api_version=api_version,
            deployment=self._deployment,
            api_key=api_key,
            authentication_method=authentication_method,
            # Prefer None over empty strings for AAD fields
            client_id=client_id or None,
            client_secret=client_secret or None,
            tenant_id=tenant_id or None,
        )

    async def generate_response(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: list[ChatCompletionToolParam] | None = None,
        tool_choice: ChatCompletionToolChoiceOptionParam | None = None,
        json_mode: bool = False,
    ) -> ChatCompletionMessage:
        """
        Generate a non-streaming response using Chat Completions.
        """
        logger.debug(
            "Generating OpenAI response",
            model=self.model,
            deployment=self._deployment,
            message_count=len(messages),
            has_tools=tools is not None,
            json_mode=json_mode,
        )
        try:
            effective_tool_choice: Any = (
                tool_choice
                if tool_choice is not None
                else ("auto" if tools else NOT_GIVEN)
            )

            response = self.client.chat.completions.create(
                # For Azure, this MUST be the deployment name:
                model=self._deployment,
                messages=messages,
                tools=tools or NOT_GIVEN,
                tool_choice=effective_tool_choice,
                response_format={"type": "json_object"} if json_mode else NOT_GIVEN,
                temperature=0.2,
            )

            if not getattr(response, "choices", None):
                raise RuntimeError(
                    "OpenAI chat.completions.create returned a response missing 'choices' or it was empty"
                )
            return response.choices[0].message

        except BadRequestError as error:
            logger.error(
                "OpenAI API request failed",
                error_type="BadRequestError",
                error_code=getattr(error, "code", None),
                error_message=error.message,
                model=self.model,
                deployment=self._deployment,
                exc_info=True,
            )

            message = error.message
            if isinstance(error.body, dict):
                message = error.body.get("message", message)

                # Content filter path
                if (
                    getattr(error, "code", None) == "content_filter"
                    and "innererror" in error.body
                ):
                    content_filter_results = error.body["innererror"].get(
                        "content_filter_result", {}
                    )
                    raise ContentFilterError(message, content_filter_results)

                # Token limit (AOAI-style) pattern
                token_error_pattern = (
                    r"This model's maximum context length is (\d+) tokens, "
                    r"however you requested (\d+) tokens \((\d+) in your prompt; "
                    r"(\d+) for the completion\)\. Please reduce your prompt; or "
                    r"completion length\."
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
        json_mode: bool = False,
    ) -> AsyncIterator[str]:
        """
        Generate a streaming response using Chat Completions.
        Yields content chunks as they arrive.
        """
        logger.debug(
            "Generating streaming OpenAI response",
            model=self.model,
            deployment=self._deployment,
            message_count=len(messages),
            has_tools=tools is not None,
            json_mode=json_mode,
        )
        try:
            effective_tool_choice: Any = (
                tool_choice
                if tool_choice is not None
                else ("auto" if tools else NOT_GIVEN)
            )

            stream = self.client.chat.completions.create(
                model=self._deployment,  # Azure deployment name
                messages=messages,
                tools=tools or NOT_GIVEN,
                tool_choice=effective_tool_choice,
                response_format={"type": "json_object"} if json_mode else NOT_GIVEN,
                temperature=0.2,
                stream=True,
            )

            # Support both sync and async iterables to be future-proof
            if hasattr(stream, "__aiter__"):
                async for chunk in stream:  # type: ignore[unreachable]
                    if (
                        getattr(chunk, "choices", None)
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
                        yield chunk.choices[0].delta.content
            else:
                for chunk in stream:
                    if (
                        getattr(chunk, "choices", None)
                        and chunk.choices[0].delta
                        and chunk.choices[0].delta.content
                    ):
                        yield chunk.choices[0].delta.content

        except BadRequestError as error:
            logger.error(
                "OpenAI streaming API request failed",
                error_type="BadRequestError",
                error_code=getattr(error, "code", None),
                error_message=error.message,
                model=self.model,
                deployment=self._deployment,
                exc_info=True,
            )

            message = error.message
            if isinstance(error.body, dict):
                message = error.body.get("message", message)

                if (
                    getattr(error, "code", None) == "content_filter"
                    and "innererror" in error.body
                ):
                    content_filter_results = error.body["innererror"].get(
                        "content_filter_result", {}
                    )
                    raise ContentFilterError(message, content_filter_results)

                token_error_pattern = (
                    r"This model's maximum context length is (\d+) tokens, "
                    r"however you requested (\d+) tokens \((\d+) in your prompt; "
                    r"(\d+) for the completion\)\. Please reduce your prompt; or "
                    r"completion length\."
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
