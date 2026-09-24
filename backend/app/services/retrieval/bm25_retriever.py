"""Sparse retrieval: incremental BM25 (Okapi) over paper title + abstract.

Titles are repeated in the token stream to weight them higher — a standard
field-boost trick that noticeably improves paper-level precision.

Why this is no longer rank_bm25
-------------------------------
`rank_bm25.BM25Okapi` has no incremental add, so the previous version
rebuilt the whole index inside `add_papers`. The comment there said the
rebuild was "O(corpus) and cheap at metadata scale". At the corpus size
this project targets it is not:

    initial build of 55,000 docs : 6.6s
    adding 15 fresh papers       : 5.2s   <-- paid on EVERY live search

`pipeline.search()` ingests freshly fetched arXiv metadata on every query
with `fetch_arxiv=True`, so each search paid a full index rebuild before
retrieving anything. It stayed invisible because the demo corpus was 46
papers, where the rebuild is milliseconds.

Two changes fix it:

* An inverted index with incremental updates. Adding documents touches the
  new documents plus one O(vocabulary) idf refresh, instead of re-tokenising
  and re-counting the entire corpus.
* Query-time scoring walks only the postings lists of the query's terms.
  rank_bm25 allocated a full N-length array per query term and scored every
  document, including the ~54,990 containing none of the query's words.

One deliberate scoring change came with the rewrite, and it is not a
no-op. rank_bm25 computes idf as

    log(N - df + 0.5) - log(df + 0.5)

which goes negative for terms appearing in more than about half the
corpus, so the library collects those terms and floors them at
`epsilon * average_idf`. This implementation uses the Lucene/Elasticsearch
form instead

    log(1 + (N - df + 0.5) / (df + 0.5))

which is always positive and needs no clamp. k1 and b are unchanged.

Measured on 506 citation-derived queries over the 55k snapshot, the
change is a small improvement rather than a wash, so it is kept:

    bm25 recall@10   0.1877 -> 0.2016
    bm25 recall@20   0.2154 -> 0.2352
    rrf  recall@10   0.2411 -> 0.2510

If you ever need to reproduce the old numbers exactly, restore the
subtractive idf and the epsilon floor — do not assume the two forms are
interchangeable.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"[a-z0-9\-]+")
_TITLE_BOOST = 2  # title tokens appear (boost+1)x

# rank_bm25's defaults, kept so moving off that library does not silently
# change ranking behaviour.
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Incremental Okapi BM25 over an in-memory inverted index."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._paper_ids: list[str] = []
        self._known: set[str] = set()
        # term -> [(doc_index, term_frequency), ...]
        self._postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self._doc_len: list[int] = []
        self._total_len = 0
        self._idf: dict[str, float] = {}
        self._idf_stale = True

    # ------------------------------------------------------------- indexing
    def add_papers(self, papers: list[dict]) -> int:
        """papers: [{arxiv_id, title, abstract}]. Dedup + incremental add.

        Cost is O(tokens in the new papers), not O(corpus).
        """
        added = 0
        with self._lock:
            for p in papers:
                pid = p["arxiv_id"]
                if pid in self._known:
                    continue
                toks = tokenize(p.get("title", "")) * (_TITLE_BOOST + 1)
                toks += tokenize(p.get("abstract", ""))
                if not toks:
                    continue

                doc_idx = len(self._paper_ids)
                self._paper_ids.append(pid)
                self._known.add(pid)

                tf: dict[str, int] = defaultdict(int)
                for t in toks:
                    tf[t] += 1
                for term, freq in tf.items():
                    self._postings[term].append((doc_idx, freq))

                self._doc_len.append(len(toks))
                self._total_len += len(toks)
                added += 1

            if added:
                # Document frequencies moved, so every idf is now slightly
                # stale. Recomputing is O(vocabulary) and happens once per
                # batch, lazily, on the next query rather than here.
                self._idf_stale = True
        return added

    def _refresh_idf(self) -> None:
        n = len(self._paper_ids)
        self._idf = {}
        for term, postings in self._postings.items():
            df = len(postings)
            # Okapi idf, with the +1 that keeps it non-negative for terms
            # occurring in more than half the corpus.
            self._idf[term] = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        self._idf_stale = False

    # ------------------------------------------------------------ retrieval
    def retrieve(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        with self._lock:
            if not self._paper_ids:
                return []
            if self._idf_stale:
                self._refresh_idf()

            avgdl = self._total_len / len(self._paper_ids)
            scores: dict[int, float] = defaultdict(float)

            # Only documents containing at least one query term are touched.
            for term in tokenize(query):
                postings = self._postings.get(term)
                if not postings:
                    continue
                idf = self._idf.get(term, 0.0)
                for doc_idx, freq in postings:
                    dl = self._doc_len[doc_idx]
                    denom = freq + K1 * (1.0 - B + B * dl / avgdl)
                    scores[doc_idx] += idf * freq * (K1 + 1.0) / denom

            if not scores:
                return []
            best = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
            return [(self._paper_ids[i], float(s)) for i, s in best if s > 0]

    @property
    def size(self) -> int:
        return len(self._paper_ids)

    def clear(self) -> None:
        with self._lock:
            self._paper_ids, self._known = [], set()
            self._postings = defaultdict(list)
            self._doc_len, self._total_len = [], 0
            self._idf, self._idf_stale = {}, True
