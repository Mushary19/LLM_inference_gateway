"""HTTP schemas for POST /v1/chat/completions (a subset of the OpenAI chat format).

Unknown request fields are ignored, so existing OpenAI client SDKs can call the endpoint.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.core.config import settings


class ChatMessageIn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class StreamOptions(BaseModel):
    # Send a final chunk with token usage (null when the provider reported none) before
    # [DONE]. Other OpenAI stream options are ignored.
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessageIn] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    stream: bool = False
    # Only used when stream is true; ignored otherwise.
    stream_options: StreamOptions | None = None

    @field_validator("max_tokens")
    @classmethod
    def _within_generation_limit(cls, value: int | None) -> int | None:
        if value is not None and value > settings.max_generation_tokens:
            raise ValueError(f"max_tokens must be <= {settings.max_generation_tokens}")
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


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    # Providers don't report creation times through the normalized interface.
    created: int = 0
    owned_by: str


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    # null when the provider didn't report token usage.
    usage: ChatUsage | None


# --- Streaming (stream=true): one ChatCompletionChunk per SSE "data:" event ---
# Serialized with exclude_unset=True, so a field is present exactly when it was set: the
# role chunk's delta is {"role": "assistant", "content": ""}, a content chunk's delta is
# {"content": "..."}, the finish chunk's delta is {}, and only the usage chunk has "usage".


class ChunkDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChunkChoice(BaseModel):
    index: int
    delta: ChunkDelta
    finish_reason: Literal["stop", "length"] | None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: list[ChunkChoice]
    # Only on the usage chunk (requested with stream_options.include_usage), which has no
    # choices; null when the provider didn't report usage.
    usage: ChatUsage | None = None


class StreamError(BaseModel):
    message: str
    type: Literal["provider_error", "server_error"]
    code: Literal["provider_timeout", "provider_unavailable", "provider_error", "internal_error"]


class StreamErrorEvent(BaseModel):
    """Sent instead of [DONE] when a stream fails after it started."""

    error: StreamError
