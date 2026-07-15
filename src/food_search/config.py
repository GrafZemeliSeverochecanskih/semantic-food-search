"""Central configuration. Secrets come from the environment (or a local .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env loader (KEY=VALUE lines); avoids an extra dependency.

    Does not override variables already set in the environment.
    """
    env_file = path or PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class Config:
    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    items_csv: Path = field(default=None) # type: ignore[assignment]
    queries_csv: Path = field(default=None) # type: ignore[assignment]

    # Retrieval
    embedder: str = "openai" # "openai" | "local"
    openai_embedding_model: str = "text-embedding-3-small"
    local_embedding_model: str = "intfloat/multilingual-e5-small"
    rrf_k: int = 60 # standard constant from the original RRF paper
    candidates_per_channel: int = 50 # depth of each retrieval channel before fusion
    top_k: int = 10 # results returned per query

    # LLM (rerank + judge)
    rerank_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"
    rerank_pool: int = 20 # candidates passed to the LLM reranker

    def __post_init__(self) -> None:
        if self.items_csv is None:
            self.items_csv = self.data_dir / "5k_items_curated.csv"
        if self.queries_csv is None:
            self.queries_csv = self.data_dir / "queries.csv"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        load_dotenv() # so OPENAI_API_KEY is set before any client is built
