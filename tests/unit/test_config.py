"""Settings load correctly and secrets do not leak into logs."""

from paddock.config import Settings


def test_defaults_are_usable_without_any_env() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.embedding_dim == 1024
    assert settings.hkjc_request_delay_s >= 1.0, "politeness delay must not drop below 1 req/s"
    assert settings.hkjc_base_url.startswith("https://")


def test_judge_provider_differs_from_generator_by_default() -> None:
    """A model grading its own output shows self-preference bias (spec §9)."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.judge_provider != settings.llm_provider


def test_secrets_are_not_exposed_by_repr() -> None:
    settings = Settings(_env_file=None, deepseek_api_key="sk-should-not-appear")  # type: ignore[call-arg]

    assert "sk-should-not-appear" not in repr(settings)
    assert settings.deepseek_api_key is not None
    assert settings.deepseek_api_key.get_secret_value() == "sk-should-not-appear"


def test_the_ui_points_at_a_local_api_by_default() -> None:
    """The demo and the API are two processes. A default that is not local would
    make a fresh clone talk to somewhere else, or to nowhere."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.ui_api_base_url == "http://127.0.0.1:8000"
