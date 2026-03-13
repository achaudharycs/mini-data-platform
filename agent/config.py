"""Configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent


def _default_warehouse_path() -> Path:
    env = os.environ.get("WAREHOUSE_PATH")
    if env:
        return Path(env)
    return _PROJECT_ROOT / "warehouse" / "data.duckdb"


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str = field(repr=False, default="")
    warehouse_path: Path = field(default_factory=_default_warehouse_path)
    model: str = "claude-sonnet-4-20250514"
    max_iterations: int = 15
    max_sql_retries: int = 3
    max_result_rows: int = 100
    page_size: int = 50

    @classmethod
    def from_env(cls) -> Config:
        return cls(
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            warehouse_path=_default_warehouse_path(),
            model=os.environ.get("AGENT_MODEL", "claude-sonnet-4-20250514"),
        )
