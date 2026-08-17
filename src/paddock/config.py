"""Typed application settings, loaded once from the environment.

Every tunable in paddock lives here. Nothing else in the codebase calls
``os.getenv`` — a setting that is not in this file does not exist, which means the
full configuration surface is one file long and reviewable in a minute.

``get_settings`` is cached, so the environment is read once per process and the same
object is shared. Tests override by constructing ``Settings(...)`` directly rather
than mutating the environment.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, PostgresDsn, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["anthropic", "deepseek", "gemini", "openai"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Database ────────────────────────────────────────────────────────────────
    database_url: PostgresDsn = Field(
        default=PostgresDsn("postgresql+psycopg://paddock:paddock@localhost:55432/paddock"),
    )

    # ── LLM providers ───────────────────────────────────────────────────────────
    # DeepSeek and Gemini both expose OpenAI-compatible endpoints, so all three
    # non-Anthropic providers share one client and differ only by base URL and model.
    llm_provider: LLMProvider = "deepseek"
    llm_model: str = "deepseek-chat"

    anthropic_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    # The judge must differ from the generator: a model grading its own output shows
    # self-preference bias. Enforced at runtime in the eval harness, not just by convention.
    judge_provider: LLMProvider = "gemini"
    judge_model: str = "gemini-3.6-flash"

    # ── Embeddings (local, CPU) ─────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # ── Ingestion ───────────────────────────────────────────────────────────────
    hkjc_base_url: str = "https://racing.hkjc.com"
    hkjc_request_delay_s: float = 1.0
    hkjc_user_agent: str = "paddock/0.1 (personal research project; contact via GitHub)"
    hkjc_max_retries: int = 3  # attempts in total, not retries after the first

    # Backoff between attempts, doubling each time and capped at 30s. Separate from
    # the politeness delay: one paces a healthy server, the other backs away from a
    # struggling one.
    hkjc_retry_backoff_s: float = 2.0

    # ── Observability ───────────────────────────────────────────────────────────
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ── API ─────────────────────────────────────────────────────────────────────
    api_rate_limit_per_minute: int = 10
    api_daily_llm_call_cap: int = 500

    # ── Alerting (reuses the existing Telegram bot) ─────────────────────────────
    telegram_api_token: SecretStr | None = None
    telegram_chat_id: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment on first call."""
    return Settings()
