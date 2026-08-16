"""One interface, four providers.

The project has to run on three different budgets: a paid key while developing, a
free tier on the public demo, and Anthropic credit for the eval judge. Switching
between them is a config change here, not a change anywhere else — nothing above
this module imports an SDK.

## Why there are only two clients for four providers

DeepSeek, Gemini and OpenAI all speak the OpenAI chat dialect; they differ by base
URL and model name. So they share one client, and Anthropic — which does not — gets
its own. That is the second caller that justifies the abstraction, and no more.

## Streaming is the only code path

Providers implement `stream` and nothing else; `complete` is a free function that
joins it. A provider that returned one thing and streamed another would let the
citation check pass text the reader never saw, and this shape makes that
unrepresentable rather than merely discouraged.

## Keys are checked when the client is built

A missing key should stop the process at startup with a sentence naming the variable,
not surface as a 401 three seconds into someone's first answer.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Protocol, cast, runtime_checkable

from pydantic import SecretStr

from paddock.config import LLMProvider, Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from openai.types.chat import ChatCompletionMessageParam

Role = Literal["system", "user", "assistant"]

# The three providers that speak the OpenAI dialect, and where they speak it.
_OPENAI_COMPATIBLE: dict[LLMProvider, str] = {
    "deepseek": "https://api.deepseek.com",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai": "https://api.openai.com/v1",
}


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@runtime_checkable
class LLM(Protocol):
    """What the agent needs from a model. Nothing about tools, images or embeddings."""

    name: str

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        """Yield the reply in pieces, in order."""
        ...


class OpenAICompatibleLLM:
    """DeepSeek, Gemini and OpenAI — same dialect, different base URL."""

    def __init__(self, *, name: str, model: str, api_key: str, base_url: str) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url
        self._api_key = api_key
        self._client: object | None = None

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        from openai import OpenAI

        if self._client is None:
            self._client = OpenAI(api_key=self._api_key, base_url=self.base_url)
        assert isinstance(self._client, OpenAI)

        payload = cast(
            "list[ChatCompletionMessageParam]",
            [{"role": message.role, "content": message.content} for message in messages],
        )
        response = self._client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=0.0,  # form questions have one right answer; do not improvise
            stream=True,
        )
        for piece in response:
            content = piece.choices[0].delta.content if piece.choices else None
            if content:
                yield content


class AnthropicLLM:
    """Anthropic's own dialect — system prompt is a parameter, not a message."""

    name = "anthropic"

    def __init__(self, *, model: str, api_key: str) -> None:
        self.model = model
        self.base_url = "https://api.anthropic.com"
        self._api_key = api_key
        self._client: object | None = None

    def stream(self, messages: Sequence[Message]) -> Iterator[str]:
        from anthropic import Anthropic

        if self._client is None:
            self._client = Anthropic(api_key=self._api_key)
        assert isinstance(self._client, Anthropic)

        system = "\n\n".join(m.content for m in messages if m.role == "system")
        turns = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        with self._client.messages.stream(
            model=self.model,
            max_tokens=1024,
            temperature=0.0,
            system=system,
            messages=turns,  # type: ignore[arg-type]
        ) as stream:
            yield from stream.text_stream


def complete(llm: LLM, messages: Sequence[Message]) -> str:
    """The whole reply. A function over `stream`, so there is only ever one code
    path — a provider that streamed one thing and returned another would let the
    citation check pass text the reader never saw."""
    return "".join(llm.stream(messages))


def build_llm(settings: Settings) -> OpenAICompatibleLLM | AnthropicLLM:
    """Construct the configured provider, or fail now with a usable message."""
    provider = settings.llm_provider
    key = _require_key(settings, provider)

    if provider == "anthropic":
        return AnthropicLLM(model=settings.llm_model, api_key=key)

    return OpenAICompatibleLLM(
        name=provider,
        model=settings.llm_model,
        api_key=key,
        base_url=_OPENAI_COMPATIBLE[provider],
    )


def _require_key(settings: Settings, provider: LLMProvider) -> str:
    secret: SecretStr | None = getattr(settings, f"{provider}_api_key", None)
    if secret is None:
        raise ValueError(
            f"{provider.upper()}_API_KEY is not set, but LLM_PROVIDER={provider}. "
            "Set it in .env, or switch provider."
        )
    return secret.get_secret_value()


@lru_cache(maxsize=1)
def get_llm() -> LLM:
    """The process-wide provider. One client, reused across requests."""
    return build_llm(get_settings())
