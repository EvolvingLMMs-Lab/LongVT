import os
import time
import uuid
from typing import Optional

import openai

from .base_server import BaseServer, ChatCompletionRequest, ChatCompletionResponse
from .mapping import register_server


@register_server("openai")
class OpenAIServer(BaseServer):
    def __init__(self, api_key: str = None, base_url: Optional[str] = None):
        super().__init__()
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
        self.base_url = base_url if base_url else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

        # Configure OpenAI client
        self.client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        # Configure async OpenAI client
        self.async_client = openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Synchronous chat completion using OpenAI API"""
        try:
            # Convert request to OpenAI format
            openai_request = {
                "model": request.model,
                "messages": request.messages,
                **request.kwargs,
            }

            # Make API call
            response = self.client.chat.completions.create(**openai_request)

            # Convert response to our format
            choices = []
            for choice in response.choices:
                choice_dict = {
                    "index": choice.index,
                    "message": {"role": choice.message.role, "content": choice.message.content},
                    "finish_reason": choice.finish_reason,
                }
                choices.append(choice_dict)

            return ChatCompletionResponse(
                id=response.id, object=response.object, created=response.created, model=response.model, choices=choices
            )

        except Exception as e:
            # Create error response
            error_choice = {
                "index": 0,
                "message": {"role": "assistant", "content": f"Error: {str(e)}"},
                "finish_reason": "error",
            }

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[error_choice],
            )

    async def chat_completion_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Asynchronous chat completion using OpenAI API"""
        try:
            # Convert request to OpenAI format
            openai_request = {
                "model": request.model,
                "messages": request.messages,
                **request.kwargs,
            }

            # Make async API call
            response = await self.async_client.chat.completions.create(**openai_request)

            # Convert response to our format
            choices = []
            for choice in response.choices:
                choice_dict = {
                    "index": choice.index,
                    "message": {"role": choice.message.role, "content": choice.message.content},
                    "finish_reason": choice.finish_reason,
                }
                choices.append(choice_dict)

            return ChatCompletionResponse(
                id=response.id, object=response.object, created=response.created, model=response.model, choices=choices
            )

        except Exception as e:
            # Create error response
            error_choice = {
                "index": 0,
                "message": {"role": "assistant", "content": f"Error: {str(e)}"},
                "finish_reason": "error",
            }

            return ChatCompletionResponse(
                id=str(uuid.uuid4()),
                object="chat.completion",
                created=int(time.time()),
                model=request.model,
                choices=[error_choice],
            )

    def get_available_models(self) -> list[str]:
        """Get list of available models from OpenAI"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception:
            # Return common OpenAI models if API call fails
            return ["gpt-4", "gpt-4-turbo-preview", "gpt-4-vision-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"]
