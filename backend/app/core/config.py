"""Central application configuration.

Every tunable in the pipeline (models, fusion weights, chunk sizes, DB URL,
LLM backend) is controlled here and overridable via environment variables or
a `.env` file, e.g. ``EMBEDDING_MODEL=BAAI/bge-base-en-v1.5``.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ------------------------------------------------------------------ app
    app_name: str = "arXiv Semantic Search Engine"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # ------------------------------------------------------------ database
    # SQLite by default (zero-setup); point at PostgreSQL in production:
    #   DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/arxiv_rag
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'data' / 'app.db'}"

    # ------------------------------------------------------------ storage
    data_dir: Path = PROJECT_ROOT / "data"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"
    index_dir: Path = PROJECT_ROOT / "data" / "indexes"

    # ------------------------------------------------------------- arXiv
    # Live candidates pulled per query. Kept small on purpose: this is a
    # synchronous call inside /search, and 15 fresh candidates per query
    # accumulate quickly across searches (each also costs an embedding).
    arxiv_max_results: int = 15
    arxiv_api_url: str = "https://export.arxiv.org/api/query"
    # Hard cap on the live-fetch HTTP call. Search must never hang on
    # a slow/blocked route to the arXiv API — on timeout it degrades to
    # the local corpus and reports why.
    arxiv_timeout_s: float = 8.0
    # Politeness controls (arXiv asks for ~1 request / 3s): identical
    # queries are cached, and uncached fetches are spaced out.
    arxiv_min_interval_s: float = 3.0
    arxiv_cache_ttl_s: float = 300.0
    # Cold-start seeding: when the corpus is empty at startup, ingest
    # the bundled seed corpus (~45 canonical IR/RAG/NLP papers) so the
    # first search always has something to rank — no dependence on the
    # live arXiv API being reachable. Abstracts in the seed file are
    # condensed summaries; full texts are fetched on QA selection.
    auto_seed_if_empty: bool = True
    seed_file: str = ""   # empty -> PROJECT_ROOT/seed/arxiv_seed.jsonl
    arxiv_categories: str = "cs.AI,cs.CL,cs.LG,cs.CV,cs.IR,cs.DC"

    # -------------------------------------------------------- embeddings
    # backend: "sentence-transformers" (production) | "hash" (dependency-free
    # deterministic fallback used for tests / environments without torch)
    embedding_backend: str = "sentence-transformers"
    # all-MiniLM-L6-v2 is a 2021 general-purpose encoder trained on web/QA
    # pairs; scientific abstracts are out of distribution for it. Anything in
    # the BGE / E5 / GTE / SPECTER families is a better fit for this corpus --
    # measure the swap with `python -m evaluation.harness --model ...` rather
    # than taking that on faith.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384             # used by the hash fallback; ST models
    embedding_batch_size: int = 64       # report their own dim at runtime.
    # Asymmetric encoders expect a role prefix on each side. Left empty the
    # model still runs and silently gives up a large slice of its advantage,
    # which reads as "the better model is worse" and gets it reverted.
    # Empty string = auto-detect from the model name (see embeddings.py).
    embedding_query_prefix: str = "auto"
    embedding_doc_prefix: str = "auto"

    # ---------------------------------------------------------- retrieval
    retrieval_top_k: int = 30            # candidates from each retriever
    fusion_method: str = "rrf"           # "rrf" | "weighted"
    fusion_alpha: float = 0.5            # dense weight for "weighted" fusion
    dynamic_alpha: bool = True           # per-query alpha tuning (DAT-lite)
    rrf_k: int = 60

    # --------------------------------------------------------- reranking
    # Candidate pool handed to the reranker = rerank_top_k * this factor
    # (floored at retrieval_top_k). Reranking only helps if it is given more
    # candidates than it is asked to return.
    rerank_pool_factor: int = 5
    reranker_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10               # papers surviving the reranker
    # The cross-encoder reads (query, title+abstract) pairs. Cost is roughly
    # linear in this window, so shortening it is tempting -- but measured on
    # 60 citation-derived queries over the 55k snapshot it is a real trade,
    # not a free win:
    #
    #   512 tokens  MRR 0.1564  R@5 0.1833  2463ms / 30 candidates
    #   256 tokens  MRR 0.1472  R@5 0.1667  1352ms / 30 candidates
    #
    # 1.8x faster for -6% MRR and -9% R@5. Default stays at 512: an accuracy
    # cut should be chosen deliberately, not inherited from a speed tweak.
    # Drop to 256 only if latency is the binding constraint and you are
    # willing to state the cost.
    rerank_max_length: int = 512
    # Corrective retrieval: if the best reranked score is below this
    # confidence, expand the query and retry once (Corrective-RAG style).
    #
    # DISABLED BY DEFAULT because the threshold was never calibrated against
    # the score distribution it is compared to. ms-marco-MiniLM emits logits
    # that sigmoid to near zero on scientific abstracts: measured over 15
    # real queries the top score had median 0.0003 and max 0.0257, against a
    # threshold of 0.15. The gate therefore failed on 15/15 queries and
    # every search paid a second full retrieve+rerank -- a 2x latency tax
    # that fired unconditionally rather than on low confidence.
    #
    # Re-enabling needs a threshold derived from the measured distribution
    # (or a relative test -- is the top result clearly better than the rest
    # -- rather than an absolute one), not another guess.
    corrective_retrieval: bool = False
    corrective_score_threshold: float = 0.15

    # ---------------------------------------------------------- chunking
    chunk_size: int = 1000               # characters per chunk (target)
    chunk_overlap: int = 150
    min_chunk_chars: int = 200
    # Papers auto-selected (and only then downloaded/indexed) when
    # Multi-Paper QA is asked without an explicit paper list. Search
    # itself never downloads PDFs.
    lazy_load_top_n: int = 3

    # ----------------------------------------------------------------- QA
    qa_chunk_top_k: int = 8              # chunks fed to the LLM per answer
    map_reduce_per_doc_chunks: int = 4   # chunks per paper in map stage
    # llm backend:
    #   "auto"              — pick the strongest available at startup:
    #                         openai-compatible endpoint (if reachable)
    #                         -> local HF model (if torch+transformers)
    #                         -> extractive
    #   "extractive"        — citation-grounded sentence synthesis, no model
    #   "hf"                — local HuggingFace model (llm_model)
    #   "openai-compatible" — Ollama / vLLM / LM Studio / OpenAI-style API
    llm_backend: str = "auto"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"        # local HF default
    llm_api_model: str = "llama3.1:8b"                 # served-model name (Ollama etc.)
    llm_api_base: str = "http://localhost:11434/v1"    # Ollama default
    llm_api_key: str = "not-needed"
    llm_max_new_tokens: int = 700
    llm_temperature: float = 0.2
    # Grounding guard: post-validate generated answers — strip citations
    # pointing at non-existent sources and regenerate extractively when a
    # generative answer carries no valid citations at all (unsupported).
    llm_grounding_check: bool = True

    # -------------------------------------------------------------- misc
    max_conversation_turns: int = 6      # follow-up context window (turns)
    request_timeout_s: int = 60


settings = Settings()
settings.pdf_dir.mkdir(parents=True, exist_ok=True)
settings.index_dir.mkdir(parents=True, exist_ok=True)
