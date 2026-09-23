"""Embedding generation.

Production backend: SentenceTransformers (all-MiniLM-L6-v2 by default;
swap to BGE etc. via EMBEDDING_MODEL). A deterministic hashed n-gram
fallback keeps the whole pipeline runnable — and testable — on machines
without torch or network access to model hubs. Both produce L2-normalized
vectors so FAISS inner product == cosine similarity.
"""
from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache

import numpy as np

from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class BaseEmbedder:
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def encode_query(self, text: str) -> np.ndarray:
        """Query-side encoding. Symmetric by default; overridden where the
        model expects a distinct query prefix."""
        return self.encode_one(text)


# Asymmetric retrieval encoders are trained with a role prefix on each side:
# the query and the document go through the same weights but are told which
# they are. Omitting the prefix does not error -- it just quietly costs a
# chunk of the model's retrieval quality, which is how a team concludes that
# a stronger model performed worse and reverts to MiniLM.
_PREFIX_BY_MODEL: dict[str, tuple[str, str]] = {
    "intfloat/e5": ("query: ", "passage: "),
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "BAAI/bge": (
        "Represent this sentence for searching relevant passages: ", ""
    ),
    # GTE, SPECTER and the sentence-transformers/* family are symmetric.
    "thenlper/gte": ("", ""),
    "allenai/specter": ("", ""),
    "sentence-transformers/": ("", ""),
}


def auto_prefixes(model_name: str) -> tuple[str, str]:
    for stem, pair in _PREFIX_BY_MODEL.items():
        if model_name.lower().startswith(stem.lower()):
            return pair
    return ("", "")


class SentenceTransformerEmbedder(BaseEmbedder):
    """Encodes documents and queries with the model's expected role prefixes.

    `encode()` is the document path (it is what the indexers call);
    `encode_query()` is the query path. They differ only by prefix, but that
    difference is the whole point of an asymmetric encoder.
    """

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        auto_q, auto_d = auto_prefixes(model_name)
        self.q_prefix = (
            auto_q if settings.embedding_query_prefix == "auto"
            else settings.embedding_query_prefix
        )
        self.d_prefix = (
            auto_d if settings.embedding_doc_prefix == "auto"
            else settings.embedding_doc_prefix
        )
        logger.info(
            "Loaded embedding model %s (dim=%d, query_prefix=%r, doc_prefix=%r)",
            model_name, self.dim, self.q_prefix, self.d_prefix,
        )

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        vecs = self.model.encode(
            texts,
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")

    def encode(self, texts: list[str]) -> np.ndarray:
        """Document side. Used by every indexing path."""
        return self._encode(texts, self.d_prefix)

    def encode_query(self, text: str) -> np.ndarray:
        """Query side. Used by retrieval."""
        return self._encode([text], self.q_prefix)[0]


class HashEmbedder(BaseEmbedder):
    """Deterministic bag-of-hashed-ngrams embedding (no ML dependencies).

    Not a semantic model — it captures lexical/character overlap — but it is
    stable, fast, and lets the retrieval/QA plumbing run end-to-end where
    torch is unavailable. Never use in production; set
    EMBEDDING_BACKEND=sentence-transformers there.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    @staticmethod
    def _tokens(text: str) -> list[str]:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        grams = list(toks)
        grams += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        for t in toks:
            if len(t) >= 5:
                grams += [t[i : i + 4] for i in range(len(t) - 3)]
        return grams

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for g in self._tokens(text):
                h = int.from_bytes(
                    hashlib.blake2b(g.encode(), digest_size=8).digest(), "little"
                )
                idx = h % self.dim
                sign = 1.0 if (h >> 63) & 1 else -1.0
                out[i, idx] += sign
            n = np.linalg.norm(out[i])
            if n > 0:
                out[i] /= n
        return out


@lru_cache(maxsize=1)
def get_embedder() -> BaseEmbedder:
    if settings.embedding_backend == "sentence-transformers":
        try:
            return SentenceTransformerEmbedder(settings.embedding_model)
        except Exception as exc:
            logger.warning(
                "sentence-transformers unavailable (%s); falling back to "
                "hash embedder — retrieval quality will be lexical only.",
                exc,
            )
    return HashEmbedder(settings.embedding_dim)
