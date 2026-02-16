"""Centralized settings management using environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _get_env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with default."""
    return int(os.getenv(key, str(default)))


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable with default."""
    return float(os.getenv(key, str(default)))


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable with default."""
    val = os.getenv(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


@dataclass
class APISettings:
    """API server settings."""
    host: str = _get_env("GREENEDGE_HOST", "127.0.0.1")
    port: int = _get_env_int("GREENEDGE_PORT", 8000)
    confidence_threshold: float = _get_env_float("GREENEDGE_CONFIDENCE_THRESHOLD", 0.55)
    api_key: str | None = _get_env("GREENEDGE_API_KEY", "") or None
    rate_limit_per_minute: int = _get_env_int("GREENEDGE_RATE_LIMIT_PER_MINUTE", 60)


@dataclass
class SimulatorSettings:
    """Simulator settings."""
    sla_ms: float = _get_env_float("GREENEDGE_SLA_MS", 120.0)
    episode_length: int = _get_env_int("GREENEDGE_EPISODE_LENGTH", 50)
    default_seed: int = _get_env_int("GREENEDGE_DEFAULT_SEED", 42)


@dataclass
class LoggingSettings:
    """Logging settings."""
    level: str = _get_env("GREENEDGE_LOG_LEVEL", "INFO")


@dataclass
class Settings:
    """Root settings container."""
    api: APISettings
    simulator: SimulatorSettings
    logging: LoggingSettings

    @classmethod
    def load(cls) -> Settings:
        """Load settings from environment."""
        # Try to load from .env file if it exists
        try:
            from dotenv import load_dotenv
            env_path = Path(__file__).parent.parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        except ImportError:
            pass  # python-dotenv not installed, use os.environ only

        return cls(
            api=APISettings(),
            simulator=SimulatorSettings(),
            logging=LoggingSettings(),
        )


# Global settings instance
settings = Settings.load()
