"""The provider seam — construction and message mapping only, no network.

Nothing here calls an API. What is worth pinning is that choosing a provider is a
config change rather than a code change, that a missing key fails at startup with a
sentence a human can act on rather than a 401 mid-answer, and that `complete()` and
`stream()` cannot drift apart.
"""

from __future__ import annotations

import pytest
from tests.doubles import ScriptedLLM

from paddock.config import Settings
from paddock.llm.provider import LLM, Message, build_llm, complete

_ANY_KEY = "sk-test-not-a-real-key"


def test_the_scripted_double_satisfies_the_protocol() -> None:
    """If this stops type-checking, every agent test is testing a fiction."""
    llm: LLM = ScriptedLLM("hello")

    assert complete(llm, [Message(role="user", content="hi")]) == "hello"


def test_complete_is_the_stream_joined() -> None:
    """One implementation, two shapes. A provider that streamed one thing and
    returned another would let the citation check pass text the reader never saw —
    so `complete` is a function over `stream`, not a second method to keep in sync."""
    llm = ScriptedLLM("it was hampered near the 800")

    assert complete(llm, [Message(role="user", content="?")]) == "it was hampered near the 800"


def test_the_double_streams_in_pieces() -> None:
    """Whatever the API does with chunk boundaries, the join has to be lossless."""
    llm = ScriptedLLM("it was hampered")

    assert list(llm.stream([Message(role="user", content="?")])) == ["it ", "was ", "hampered"]


def test_an_openai_compatible_provider_is_chosen_by_config() -> None:
    llm = build_llm(
        Settings(llm_provider="deepseek", llm_model="deepseek-chat", deepseek_api_key=_ANY_KEY)
    )

    assert llm.name == "deepseek"


def test_gemini_speaks_the_openai_dialect_at_its_own_url() -> None:
    """Three of the four providers share one client and differ only by base URL —
    the reason there is no per-provider class."""
    llm = build_llm(Settings(llm_provider="gemini", gemini_api_key=_ANY_KEY))

    assert "generativelanguage" in llm.base_url


def test_anthropic_gets_its_own_client() -> None:
    llm = build_llm(Settings(llm_provider="anthropic", anthropic_api_key=_ANY_KEY))

    assert llm.name == "anthropic"


def test_a_missing_key_fails_at_startup_with_a_usable_message() -> None:
    """Not at the first question, and not as a 401 three seconds into a stream."""
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        build_llm(Settings(llm_provider="deepseek", deepseek_api_key=None))
