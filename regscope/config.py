"""Configuration loading for RegScope."""

import logging
import os
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "api": {
        "api_key": "",
        "requests_per_hour": 900,
        "max_retries": 3,
        "retry_backoff_base": 2.0,
    },
    "data": {
        "data_dir": "~/.regscope/data",
        "attachments_dir": "~/.regscope/attachments",
    },
    "embedding": {
        "model": "all-MiniLM-L6-v2",
        "batch_size": 64,
    },
    "dedup": {
        "near_duplicate_threshold": 0.85,
        "num_perm": 128,
        "semantic_threshold": 0.92,
    },
    "topics": {
        "min_topic_size": 10,
        "nr_topics": "auto",
        "top_n_words": 10,
        "umap_n_neighbors": 15,
        "umap_n_components": 5,
        "umap_min_dist": 0.0,
        "hdbscan_min_cluster_size": 10,
        "hdbscan_min_samples": 5,
    },
    "classification": {
        "stance_model": "facebook/bart-large-mnli",
        "stance_confidence_threshold": 0.4,
    },
    "substantiveness": {
        "weight_length": 0.15,
        "weight_citations": 0.20,
        "weight_section_references": 0.20,
        "weight_technical_vocab": 0.15,
        "weight_data_statistics": 0.15,
        "weight_legal_arguments": 0.10,
        "weight_not_form_letter": 0.05,
        "length_optimal": 2000,
        "length_max_score": 15,
    },
    "llm": {
        "enabled": True,
        "model": "gpt-4o-mini",
    },
    "report": {
        "top_substantive_count": 20,
        "max_excerpt_length": 500,
    },
    "logging": {
        "level": "INFO",
        "log_file": "~/.regscope/regscope.log",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override dict into base dict recursively."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def find_config_file() -> Path | None:
    """Search for config.toml in standard locations."""
    candidates = [
        Path.cwd() / "config.toml",
        Path.home() / ".regscope" / "config.toml",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from TOML file, merged with defaults.

    Args:
        config_path: Explicit path to config file. If None, searches standard locations.

    Returns:
        Configuration dictionary with all settings.
    """
    config = DEFAULTS.copy()

    if config_path is not None:
        path = Path(config_path)
    else:
        path = find_config_file()

    if path is not None and path.exists():
        logger.debug("Loading config from %s", path)
        with open(path, "rb") as f:
            user_config = tomllib.load(f)
        config = _deep_merge(config, user_config)

    # Apply environment variable overrides
    env_key = os.environ.get("REGSCOPE_API_KEY")
    if env_key:
        config["api"]["api_key"] = env_key

    # Expand ~ in paths
    for section in ("data", "logging"):
        if section in config:
            for key, value in config[section].items():
                if isinstance(value, str) and "~" in value:
                    config[section][key] = str(Path(value).expanduser())

    return config


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging based on config settings."""
    log_cfg = config.get("logging", {})
    level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    log_file = log_cfg.get("log_file", "")
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
