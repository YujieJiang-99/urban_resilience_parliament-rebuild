"""Environment configuration for OpenAI-compatible LLM backends."""

from dataclasses import dataclass
import os
from pathlib import Path


DEFAULT_BASE_URL = "https://api.zhizengzeng.com/v1"
DEFAULT_MODEL = "gpt-5-nano"


@dataclass(frozen=True)
class LLMConfig:
    """Runtime configuration for an OpenAI-compatible chat endpoint."""

    base_url: str
    api_key: str
    model: str
    timeout: float
    debug: bool
    log_dir: Path


def load_llm_config(env_path: str | Path = ".env") -> LLMConfig:
    """Load LLM config from process env and a local .env file."""

    file_values = _read_env_file(env_path)
    base_url = _get_value("LLM_BASE_URL", file_values, DEFAULT_BASE_URL).rstrip("/")
    api_key = _get_value("LLM_API_KEY", file_values, "")
    model = _get_value("LLM_MODEL", file_values, DEFAULT_MODEL)
    timeout = float(_get_value("LLM_TIMEOUT", file_values, "60"))
    debug = _get_value("LLM_DEBUG", file_values, "0") == "1"
    log_dir = Path(_get_value("LLM_LOG_DIR", file_values, "./logs"))
    return LLMConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        timeout=timeout,
        debug=debug,
        log_dir=log_dir,
    )


def _get_value(key: str, file_values: dict[str, str], default: str) -> str:
    return os.environ.get(key) or file_values.get(key) or default


def _read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
