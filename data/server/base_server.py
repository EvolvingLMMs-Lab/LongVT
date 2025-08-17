from abc import ABC, abstractmethod
from typing import Any


class Message:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
        }


class ChatCompletionRequest:
    def __init__(self, model: str, messages: list[Message], **kwargs) -> None:
        self.model = model
        self.messages = messages
        self.kwargs = kwargs

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": [message.to_dict() for message in self.messages],
            **self.kwargs,
        }


class ChatCompletionResponse:
    def __init__(self, id: str, object: str, created: int, model: str, choices: list[dict[str, Any]]) -> None:
        self.id = id
        self.object = object
        self.created = created
        self.model = model
        self.choices = choices

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": self.choices,
        }


class BaseServer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        pass

    @abstractmethod
    def get_available_models(self) -> list[str]:
        pass

    @abstractmethod
    async def chat_completion_async(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        pass
