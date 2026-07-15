"""Pluggable embedding backends with on-disk caching.

Two backends behind one interface:
  * OpenAIEmbedder - text-embedding-3-small (primary; strong multilingual, ~$0.02
    to embed the whole 5k catalog).
  * LocalE5Embedder - intfloat/multilingual-e5-small via sentence-transformers
    (free/offline; also serves as an independent comparison system).

Catalog embeddings are cached in artifacts/ keyed by a fingerprint of the
model name + document texts, so re-runs cost nothing.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        """Return L2-normalized embeddings, shape (len(texts), dim)."""
        ...


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    # unit vectors -> dot product doubles as cosine in the dense index
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, 1e-12, None)


class OpenAIEmbedder:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        batch_size: int = 512,
    ):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add the key, "
                "or use --embedder local."
            )
        self.client = OpenAI()
        self.model = model
        self.batch_size = batch_size
        self.name = f"openai:{model}"

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        import time

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            # the API rejects empty strings, and newlines are best stripped
            batch = [t.replace("\n", " ") or " " for t in texts[start : start + self.batch_size]]
            for attempt in range(4): # backoff 1s/2s/4s on transient failures
                try:
                    response = self.client.embeddings.create(model=self.model, input=batch)
                    break
                except Exception: # noqa: BLE001 - transient network/rate errors
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
            vectors.extend(d.embedding for d in response.data)
        return _l2_normalize(np.asarray(vectors, dtype=np.float32))


class LocalE5Embedder:
    """multilingual-e5 requires 'query: ' / 'passage: ' prefixes."""

    def __init__(
        self,
        model: str = "intfloat/multilingual-e5-small",
        batch_size: int = 64,
    ):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model)
        self.batch_size = batch_size
        self.name = f"local:{model.split('/')[-1]}"

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        prefixed = [f"{kind}: {t}" for t in texts]
        mat = self.model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(mat, dtype=np.float32)


def make_embedder(kind: str, cfg) -> Embedder:
    if kind == "openai":
        return OpenAIEmbedder(cfg.openai_embedding_model)
    if kind == "local":
        return LocalE5Embedder(cfg.local_embedding_model)
    raise ValueError(f"Unknown embedder '{kind}' (expected 'openai' or 'local')")


class QueryCachedEmbedder:
    """Wrap an embedder with a per-text on-disk cache.

    Used on the query side: reruns and flaky networks never re-fetch a vector
    that was fetched once. Progress is monotonic across crashes.
    """

    def __init__(self, inner: Embedder, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        self._dir = cache_dir / ("qcache_" + inner.name.replace(":", "_").replace("/", "_"))
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, text: str, kind: str) -> Path:
        digest = hashlib.md5(f"{kind}\x1e{text}".encode("utf-8")).hexdigest()
        return self._dir / f"{digest}.npy"

    def embed(self, texts: list[str], kind: str = "passage") -> np.ndarray:
        # dict.fromkeys dedupes while keeping order; fetch only cache misses
        missing = [t for t in dict.fromkeys(texts) if not self._path(t, kind).exists()]
        if missing:
            mat = self.inner.embed(missing, kind=kind)
            for text, vec in zip(missing, mat):
                np.save(self._path(text, kind), vec)
        return np.stack([np.load(self._path(t, kind)) for t in texts])


def embed_corpus_cached(
    embedder: Embedder,
    texts: list[str],
    cache_dir: Path,
) -> np.ndarray:
    # key on model + full contents, so any document change invalidates the cache
    fingerprint = hashlib.md5(
        (embedder.name + "\x1e" + "\x1e".join(texts)).encode("utf-8")
    ).hexdigest()[:16]
    cache_file = cache_dir / f"emb_{embedder.name.replace(':', '_').replace('/', '_')}_{fingerprint}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    mat = embedder.embed(texts, kind="passage")
    np.save(cache_file, mat)
    return mat
