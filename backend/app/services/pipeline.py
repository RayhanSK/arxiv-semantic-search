"""End-to-end pipeline orchestrator.

Search flow (mirrors Fig 4.1/4.2/4.5 of the report) — metadata only, fast:

  query -> QueryProcessor -> [real-time arXiv metadata ingest]
        -> HybridRetriever -> Reranker -> (corrective retry) -> results

PDF download + parsing + chunk embedding are deferred ("lazy loading"):
they run only inside ask_single/ask_multi, i.e. when the user selects a
paper for QA, via the idempotent ensure_paper_indexed().
"""
from __future__ import annotations

import json
import logging
import time

from sqlalchemy import func, select

from backend.app.core.config import settings
from backend.app.db.models import Chunk, Paper, QueryLog
from backend.app.db.session import db_session, init_db
from backend.app.services.arxiv_client import PaperMeta, search_arxiv, upsert_papers
from backend.app.services.documents.chunker import chunk_document
from backend.app.services.documents.loader import load_document
from backend.app.services.embeddings import get_embedder
from backend.app.services.qa.engine import QAEngine, QAResult
from backend.app.services.query_processor import ProcessedQuery, process_query
from backend.app.services.recommender import Recommender
from backend.app.services.reranker import Reranker
from backend.app.services.retrieval.bm25_retriever import BM25Retriever
from backend.app.services.retrieval.dense_retriever import DenseRetriever
from backend.app.services.retrieval.hybrid import HybridRetriever
from backend.app.services.vector_store import FaissStore

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Singleton-ish orchestrator wired at app startup."""

    def __init__(self) -> None:
        init_db()
        self.embedder = get_embedder()
        self.paper_store = FaissStore("papers", self.embedder.dim, settings.index_dir)
        self.chunk_store = FaissStore("chunks", self.embedder.dim, settings.index_dir)
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever(self.paper_store, self.embedder)
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        self.reranker = Reranker()
        self.qa = QAEngine(self.chunk_store, self.embedder, self.reranker)
        self.recommender = Recommender(self.paper_store)
        self._warm_bm25_from_db()
        if settings.auto_seed_if_empty and self.bm25.size == 0:
            n = self.seed_from_file()
            if n:
                logger.info("Cold start: seeded %d bundled papers", n)
        self._warm_models()

    def _warm_models(self) -> None:
        """Load model weights at startup, not on the first user query.

        The embedder loads at construction, but the cross-encoder used to
        load lazily inside the first rerank call — making the first search
        pay 5–15s of model loading. A tiny forward pass through both here
        moves that cost to app startup.
        """
        try:
            self.embedder.encode_one("warmup")
            self.reranker.rerank(
                "warmup",
                [{"arxiv_id": "_", "text": "warmup", "fused_score": 1.0}],
                top_k=1,
            )
            logger.info("Models warmed (embedder + reranker)")
        except Exception:  # warming is best-effort, never fatal
            logger.debug("model warmup failed", exc_info=True)

    # ------------------------------------------------------------- warmup
    def _warm_bm25_from_db(self) -> None:
        """BM25 is in-memory; rebuild it from persisted metadata on boot."""
        with db_session() as s:
            papers = s.execute(select(Paper)).scalars().all()
            if papers:
                self.hybrid.add_papers([p.to_dict() for p in papers])
                logger.info("Warmed retrievers with %d papers from DB", len(papers))

    # ------------------------------------------------------------- ingest
    def ingest_metadata(self, metas) -> int:
        """Upsert paper metadata (from arXiv or bulk scripts) into DB+indexes."""
        with db_session() as s:
            papers = upsert_papers(s, metas)
            dicts = [p.to_dict() for p in papers]
        added = self.hybrid.add_papers(dicts)
        if added:
            self.paper_store.save()
        return added

    def seed_from_file(self, path=None) -> int:
        """Ingest the bundled offline seed corpus (JSONL of paper metadata).

        Local ingestion workflow that never touches the network — used for
        cold starts and by `scripts/ingest.py --seed`. Idempotent: existing
        arxiv_ids are skipped by the usual dedup.
        """
        from pathlib import Path

        from backend.app.core.config import PROJECT_ROOT

        path = Path(path or settings.seed_file or (PROJECT_ROOT / "seed" / "arxiv_seed.jsonl"))
        if not path.exists():
            logger.warning("seed file not found: %s", path)
            return 0
        metas: list[PaperMeta] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("arxiv_id") and rec.get("title"):
                metas.append(PaperMeta(
                    arxiv_id=rec["arxiv_id"],
                    title=rec["title"],
                    abstract=rec.get("abstract", ""),
                    authors=rec.get("authors", ""),
                    categories=rec.get("categories", ""),
                    published=rec.get("published", ""),
                    pdf_url=rec.get("pdf_url", ""),
                ))
        return self.ingest_metadata(metas)

    # ------------------------------------------------------------- search
    def search(
        self,
        query: str,
        top_k: int | None = None,
        fetch_arxiv: bool = True,
        session_id: str = "",
    ) -> dict:
        """Metadata-only search: hybrid retrieve -> rerank -> results.

        Deliberately performs NO PDF download or full-text indexing — that
        work is deferred until a paper is actually selected for QA
        (``ask_single`` / ``ask_multi``), keeping search fast regardless of
        corpus or PDF sizes.
        """
        t0 = time.perf_counter()
        top_k = top_k or settings.rerank_top_k
        pq = process_query(query)

        # 1. Real-time ingestion: pull fresh candidates from the arXiv API
        #    (metadata only — titles/abstracts, never PDFs)
        t_fetch = 0.0
        fetch_error: str | None = None
        if fetch_arxiv:
            tf = time.perf_counter()
            metas, fetch_error = search_arxiv(pq.sparse_query)
            if metas:
                self.ingest_metadata(metas)
            t_fetch = (time.perf_counter() - tf) * 1000

        # 2-4. hybrid retrieve -> rerank -> corrective retry (local only:
        # the fetch above already ingested fresh candidates, so the retry
        # re-queries our own indexes with an expanded query instead of
        # paying a second arXiv roundtrip)
        tr = time.perf_counter()
        ranked = self._retrieve_and_rerank(pq, top_k)
        if (
            settings.corrective_retrieval
            and not Reranker.confident(ranked)
            and pq.expansions == []
        ):
            logger.info("Low retrieval confidence; corrective re-query")
            pq2 = process_query(query, expand=True)
            pq2.sparse_query = f"{pq2.sparse_query} {' '.join(pq2.keywords[:5])}"
            retry = self._retrieve_and_rerank(pq2, top_k)
            if retry and (
                not ranked
                or retry[0]["rerank_score"] > ranked[0]["rerank_score"]
            ):
                ranked = retry
        t_retrieve = (time.perf_counter() - tr) * 1000

        latency = (time.perf_counter() - t0) * 1000
        self._log(session_id, "search", query, "", [r["arxiv_id"] for r in ranked], latency)
        notice = None
        if fetch_error:
            notice = f"Live arXiv fetch skipped — {fetch_error}. "
            notice += (
                "The local corpus is empty, so there is nothing to search yet: "
                "check your network/proxy access to export.arxiv.org, or bulk-"
                "ingest papers with scripts/ingest.py."
                if self.bm25.size == 0
                else "Showing results from the local corpus only."
            )
        elif not ranked and self.bm25.size == 0:
            notice = (
                "The local corpus is empty and the live fetch returned no "
                "matches — try a broader query, or ingest papers with "
                "scripts/ingest.py."
            )
        return {
            "notice": notice,
            "query": pq.normalized,
            "expansions": pq.expansions,
            "alpha_hint": round(1 - pq.technicality, 3),
            "latency_ms": round(latency, 1),
            "timings_ms": {
                "arxiv_fetch": round(t_fetch, 1),
                "retrieve_rerank": round(t_retrieve, 1),
            },
            "results": [
                {
                    "arxiv_id": r["arxiv_id"],
                    "title": r["title"],
                    "abstract": r["abstract"],
                    "authors": r.get("authors", ""),
                    "categories": r.get("categories", ""),
                    "published": r.get("published", ""),
                    "pdf_status": r.get("pdf_status", "none"),
                    "score": round(r["rerank_score"], 4),
                }
                for r in ranked
            ],
        }

    def _retrieve_and_rerank(self, pq: ProcessedQuery, top_k: int) -> list[dict]:
        # The candidate pool must be wider than the requested top_k or the
        # reranker has nothing to reorder. This previously called
        # hybrid.retrieve(pq) with no top_k at all, so the pool was always the
        # global default (30) no matter what the caller asked for -- silently
        # capping recall@k for every k > 30 during evaluation.
        pool = max(top_k * settings.rerank_pool_factor, settings.retrieval_top_k)
        fused = self.hybrid.retrieve(pq, top_k=pool)
        if not fused:
            return []
        ids = [pid for pid, _ in fused]
        scores = dict(fused)
        with db_session() as s:
            rows = s.execute(select(Paper).where(Paper.arxiv_id.in_(ids))).scalars().all()
        by_id = {p.arxiv_id: p for p in rows}
        candidates = []
        for pid in ids:
            p = by_id.get(pid)
            if not p:
                continue
            candidates.append(
                {
                    **p.to_dict(),
                    "text": f"{p.title}. {p.abstract}",
                    "fused_score": scores[pid],
                }
            )
        return self.reranker.rerank(pq.normalized, candidates, top_k=top_k)

    # ---------------------------------------------------------- lazy load
    def ensure_paper_indexed(self, arxiv_id: str) -> int:
        """Lazy pipeline for one paper: download -> parse -> chunk -> embed.

        Idempotent: returns existing chunk count if already embedded
        (no duplicate indexing NFR).
        """
        with db_session() as s:
            paper = s.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
            if paper is None:
                raise ValueError(f"unknown paper {arxiv_id}")
            if paper.pdf_status == "embedded" and paper.n_chunks > 0:
                return paper.n_chunks

            doc = load_document(paper)
            chunks = chunk_document(doc)
            if not chunks:
                paper.pdf_status = "failed"
                return 0

            texts = [c.text for c in chunks]
            vecs = self.embedder.encode(texts)
            metas = [
                {
                    "arxiv_id": paper.arxiv_id,
                    "title": paper.title,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "text": c.text,
                }
                for c in chunks
            ]
            faiss_ids = self.chunk_store.add(vecs, metas)
            # replace any stale chunk rows, then persist provenance
            s.query(Chunk).filter(Chunk.paper_id == paper.id).delete()
            for c, fid in zip(chunks, faiss_ids):
                s.add(
                    Chunk(
                        paper_id=paper.id,
                        chunk_index=c.chunk_index,
                        section=c.section,
                        text=c.text,
                        faiss_id=fid,
                        embedded=True,
                    )
                )
            paper.n_chunks = len(chunks)
            paper.pdf_status = "embedded"
        self.chunk_store.save()
        logger.info("Indexed %s: %d chunks (%s)", arxiv_id, len(chunks), doc.source)
        return len(chunks)

    # ---------------------------------------------------------------- QA
    def ask_single(
        self, question: str, arxiv_id: str,
        history: list[dict] | None = None, session_id: str = "",
    ) -> QAResult:
        t0 = time.perf_counter()
        self.ensure_paper_indexed(arxiv_id)
        with db_session() as s:
            p = s.execute(
                select(Paper).where(Paper.arxiv_id == arxiv_id)
            ).scalar_one_or_none()
        res = self.qa.answer_single(
            question, arxiv_id, title=p.title if p else "", history=history
        )
        self._log(
            session_id, "qa_single", question, res.answer, res.papers,
            (time.perf_counter() - t0) * 1000,
        )
        return res

    def ask_multi(
        self, question: str, paper_ids: list[str] | None = None,
        history: list[dict] | None = None, session_id: str = "",
    ) -> QAResult:
        t0 = time.perf_counter()
        if not paper_ids:
            # retrieval decides which papers matter for this question;
            # only those selected papers get downloaded/indexed (below)
            hit = self.search(question, fetch_arxiv=False)
            paper_ids = [r["arxiv_id"] for r in hit["results"][: settings.lazy_load_top_n]]
        for pid in paper_ids:
            try:
                self.ensure_paper_indexed(pid)
            except Exception as exc:
                logger.warning("skip %s: %s", pid, exc)
        titles: dict[str, str] = {}
        paper_status: dict[str, str] = {}
        with db_session() as s:
            for pid in paper_ids:
                p = s.execute(
                    select(Paper).where(Paper.arxiv_id == pid)
                ).scalar_one_or_none()
                if p:
                    titles[pid] = p.title
                    paper_status[pid] = p.pdf_status + (
                        f" ({p.parse_error})" if p.parse_error else ""
                    )
                else:
                    paper_status[pid] = "unknown paper id"
        res = self.qa.answer_multi(
            question, paper_ids, titles=titles, history=history,
            paper_status=paper_status,
        )
        self._log(
            session_id, "qa_multi", question, res.answer, res.papers,
            (time.perf_counter() - t0) * 1000,
        )
        return res

    # -------------------------------------------------------------- admin
    def stats(self) -> dict:
        with db_session() as s:
            n_papers = s.execute(select(func.count(Paper.id))).scalar() or 0
            n_chunks = s.execute(select(func.count(Chunk.id))).scalar() or 0
            n_queries = s.execute(select(func.count(QueryLog.id))).scalar() or 0
        from backend.app.services.llm.generator import get_llm

        llm = get_llm()
        llm_model = (
            "n/a (grounded sentence synthesis)"
            if llm.name == "extractive"
            else (settings.llm_api_model or settings.llm_model)
            if llm.name == "openai-compatible"
            else settings.llm_model
        )
        return {
            "papers": n_papers,
            "chunks": n_chunks,
            "queries": n_queries,
            "bm25_docs": self.bm25.size,
            "paper_vectors": self.paper_store.ntotal,
            "chunk_vectors": self.chunk_store.ntotal,
            "embedding_backend": type(self.embedder).__name__,
            "embedding_dim": self.embedder.dim,
            "llm_backend": llm.name
            + (" (auto-selected)" if settings.llm_backend == "auto" else ""),
            "llm_model": llm_model,
        }

    def rebuild_indexes(self) -> dict:
        """Admin: rebuild vector + BM25 indexes from the metadata DB."""
        self.paper_store.clear()
        self.chunk_store.clear()
        self.bm25.clear()
        with db_session() as s:
            papers = s.execute(select(Paper)).scalars().all()
            dicts = [p.to_dict() for p in papers]
            for p in papers:  # chunks will be lazily re-embedded on demand
                p.pdf_status = "none" if p.pdf_status != "failed" else "failed"
                p.n_chunks = 0
            s.query(Chunk).delete()
        self.hybrid.add_papers(dicts)
        self.paper_store.save()
        self.paper_store.maybe_train_ivf()
        return self.stats()

    def _log(self, session_id, mode, query, answer, papers, latency_ms) -> None:
        try:
            with db_session() as s:
                s.add(
                    QueryLog(
                        session_id=session_id, mode=mode, query=query,
                        answer=answer[:5000], sources=json.dumps(papers),
                        latency_ms=latency_ms,
                    )
                )
        except Exception:  # logging must never break a request
            logger.debug("query log write failed", exc_info=True)


_pipeline: SearchPipeline | None = None


def get_pipeline() -> SearchPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SearchPipeline()
    return _pipeline
