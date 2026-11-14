"""HTTP schemas for POST /v1/chat/completions (a subset of the OpenAI chat format).

Unknown request fields are ignored, so existing OpenAI client SDKs can call the endpoint.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class ChatMessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessageIn] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False

    @field_validator("max_tokens")
    @classmethod
    def _within_generation_limit(cls, value: int | None) -> int | None:
        if value is not None and value > settings.max_generation_tokens:
            raise ValueError(f"max_tokens must be <= {settings.max_generation_tokens}")
        return value

    @field_validator("stream")
    @classmethod
    def _streaming_not_supported_yet(cls, value: bool) -> bool:
        if value:
            raise ValueError("streaming is not supported yet")
        return value


class ChatMessageOut(BaseModel):
    role: Literal["assistant"]
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatMessageOut
    finish_reason: Literal["stop", "length"]


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: ChatUsage
