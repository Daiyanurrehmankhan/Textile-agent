"""Central place for environment configuration.

Every other module reads config from `settings` below -- nothing else in
this project should call os.getenv/os.environ directly.
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str
    test_schema: str = "test_textile"  # isolated schema tests run against, never touches dev data
    google_api_key: str | None = None  # None is fine at import time -- the agent's live tests skip
    #                                     themselves when it's unset, rather than failing
    google_model: str = "gemini-3.5-flash"
    # Backup models tried in order if google_model's call fails (rate limit, quota,
    # outage, ...). gemini-2.5-flash was dropped: it 404s on this account's API key
    # despite being listed as live on Google's docs -- not worth a dead round-trip
    # on every failover. The remaining two are confirmed live and reachable.
    google_fallback_models: tuple[str, ...] = ("gemini-3.6-flash", "gemini-3.5-flash-lite")
    # Safety cap on tool calls per agent investigation, set per investigation path rather than
    # one shared number: reasoning traces showed the mismatch path sometimes wants one more
    # verification call (e.g. a broad query_inventory({}) sweep to check whether other
    # materials show the same pattern) than a clean delay run needs. See CLAUDE.md/README for
    # the traces this is based on.
    default_max_tool_calls_delay: int = 6
    default_max_tool_calls_mismatch: int = 7


def _load() -> Settings:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set -- copy .env.example to .env and fill it in.")
    fallback_env = os.environ.get("GOOGLE_FALLBACK_MODELS")
    fallback_models = (
        tuple(m.strip() for m in fallback_env.split(",") if m.strip())
        if fallback_env
        else ("gemini-3.6-flash", "gemini-3.5-flash-lite")
    )
    return Settings(
        database_url=database_url,
        google_api_key=os.environ.get("GOOGLE_API_KEY") or None,
        google_model=os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash"),
        google_fallback_models=fallback_models,
        default_max_tool_calls_delay=int(os.environ.get("MAX_TOOL_CALLS_DELAY", "6")),
        default_max_tool_calls_mismatch=int(os.environ.get("MAX_TOOL_CALLS_MISMATCH", "7")),
    )


settings = _load()
