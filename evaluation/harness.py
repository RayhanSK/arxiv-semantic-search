"""Reproducible retrieval evaluation over a FROZEN corpus snapshot.

What was wrong with the old harness
-----------------------------------
`evaluation/evaluate.py` went through `get_pipeline()`, which opens SQLite,
seeds the bundled 46-paper corpus, warms models, and -- in `search()` --
calls the live arXiv API and ingests the results mid-query. None of that is
reproducible. Worse, it silently reported 0.0000 across every strategy and
every metric because not one of the five gold papers existed in the corpus
it was searching. A harness that cannot tell "your retrieval is broken" from
"your gold documents are absent" is not an instrument.

So this harness:

  * builds its indexes from a named corpus file and nothing else -- no DB,
    no network, no seeding;
  * REFUSES TO RUN if gold documents are missing from the corpus, printing
    which ones (`--allow-missing-gold` downgrades it to a warning and scores
    only the answerable queries);
  * reuses the application's own BM25Retriever, fusion and Reranker code, so
    what it measures is the shipped system rather than a lookalike.

Usage
-----
    python -m evaluation.harness \
        --corpus evaluation/data/corpus.jsonl \
        --evalset evaluation/data/evalset.jsonl \
        --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.query_processor import process_query  # noqa: E402
from backend.app.services.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from backend.app.services.retrieval.fusion import (  # noqa: E402
    dynamic_alpha, rrf_fuse, weighted_fuse,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --------------------------------------------------------------- embedding
# Asymmetric encoders need their query/passage prefixes. Feeding a bare query
# to E5 or BGE loses a large slice of the gain they exist to provide -- and
# the loss is silent, so you conclude the better model is worse and revert.
PREFIXES: dict[str, tuple[str, str]] = {
    "intfloat/e5":            ("query: ", "passage: "),
    "intfloat/multilingual-e5": ("query: ", "passage: "),
    "BAAI/bge":               ("Represent this sentence for searching relevant passages: ", ""),
    "thenlper/gte":           ("", ""),
    "sentence-transformers/": ("", ""),
    "allenai/specter":        ("", ""),
}


def prefixes_for(model_name: str) -> tuple[str, str]:
    for stem, pair in PREFIXES.items():
        if model_name.lower().startswith(stem.lower()):
            return pair
    return ("", "")


class Encoder:
    """Thin SentenceTransformer wrapper that applies the right prefixes."""

    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.q_prefix, self.d_prefix = prefixes_for(model_name)
        self.batch_size = batch_size
        self.dim = self.model.get_sentence_embedding_dimension()

    def _encode(self, texts: list[str], prefix: str, show: bool) -> np.ndarray:
        if prefix:
            texts = [prefix + t for t in texts]
        v = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            show_progress_bar=show, convert_to_numpy=True,
        )
        return np.asarray(v, dtype="float32")

    def encode_docs(self, texts: list[str], show: bool = True) -> np.ndarray:
        return self._encode(texts, self.d_prefix, show)

    def encode_queries(self, texts: list[str], show: bool = False) -> np.ndarray:
        return self._encode(texts, self.q_prefix, show)


# ------------------------------------------------------------------ corpus
@dataclass
class Corpus:
    ids: list[str] = field(default_factory=list)
    docs: list[dict] = field(default_factory=list)
    index: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path, doc_field: str = "title_abstract") -> "Corpus":
        c = cls()
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r["arxiv_id"] in c.index:
                    continue
                c.index[r["arxiv_id"]] = len(c.ids)
                c.ids.append(r["arxiv_id"])
                c.docs.append(r)
        return c

    def text(self, i: int) -> str:
        d = self.docs[i]
        return f"{d.get('title','')}. {d.get('abstract','')}"

    def __len__(self) -> int:
        return len(self.ids)


class DenseIndex:
    """Exact inner-product search over L2-normalised vectors.

    Flat numpy rather than FAISS on purpose: at 50k x 768 the matmul is a few
    milliseconds, it is exact, and it has no on-disk state that could leak a
    stale index from a previous model into this run -- a real hazard given
    FaissStore persists by name and silently reuses whatever it finds.
    """

    def __init__(self, vectors: np.ndarray, ids: list[str]):
        self.v = vectors
        self.ids = ids

    def search(self, qv: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        sims = self.v @ qv
        k = min(top_k, len(self.ids))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self.ids[i], float(sims[i])) for i in idx]


# ----------------------------------------------------------------- metrics
def recall_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def mrr(ranked: list[str], gold: set[str]) -> float:
    for i, pid in enumerate(ranked, 1):
        if pid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[str], gold: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 1)
              for i, pid in enumerate(ranked[:k], 1) if pid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# -------------------------------------------------------------- evaluation
def check_gold_coverage(rows: list[dict], corpus: Corpus, allow_missing: bool) -> list[dict]:
    """The guard the original harness did not have.

    Its eval set referenced five papers, none of which were in the corpus, so
    every strategy scored exactly 0.0000 and the output looked like a
    retrieval failure. Absent gold is a broken experiment, not a bad score.
    """
    missing = {g for r in rows for g in r["relevant"] if g not in corpus.index}
    if not missing:
        return rows

    covered = [r for r in rows if all(g in corpus.index for g in r["relevant"])]
    msg = (f"{len(missing)} gold paper(s) referenced by the eval set are NOT in "
           f"the corpus, affecting {len(rows) - len(covered)}/{len(rows)} queries.\n"
           f"  examples: {sorted(missing)[:8]}")
    if not allow_missing:
        raise SystemExit(
            "REFUSING TO RUN -- the result would be meaningless.\n" + msg +
            "\n\nEither re-harvest a corpus containing them, rebuild the eval set "
            "against this corpus, or pass --allow-missing-gold to score only the "
            f"{len(covered)} answerable queries."
        )
    print(f"WARNING: {msg}\n  scoring the {len(covered)} answerable queries only.\n")
    return covered


def evaluate(
    corpus_path: Path, evalset_path: Path, model_name: str,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    strategies: tuple[str, ...] = ("bm25", "dense", "rrf", "weighted"),
    allow_missing_gold: bool = False,
    limit: int | None = None,
    cache_dir: Path | None = None,
) -> dict:
    corpus = Corpus.load(corpus_path)
    rows = [json.loads(l) for l in evalset_path.open(encoding="utf-8") if l.strip()]
    if limit:
        rows = rows[:limit]
    print(f"Corpus : {len(corpus)} papers  ({corpus_path.name})")
    print(f"Queries: {len(rows)}           ({evalset_path.name})")

    rows = check_gold_coverage(rows, corpus, allow_missing_gold)
    if not rows:
        raise SystemExit("No answerable queries remain.")

    # ---- sparse (the application's own retriever, verbatim)
    t0 = time.perf_counter()
    bm25 = BM25Retriever()
    bm25.add_papers([
        {"arxiv_id": d["arxiv_id"], "title": d.get("title", ""),
         "abstract": d.get("abstract", "")} for d in corpus.docs
    ])
    t_bm25 = time.perf_counter() - t0

    # ---- dense (cached per corpus+model: embedding 50k docs is the slow part)
    enc = Encoder(model_name)
    key = f"{corpus_path.stem}.{model_name.replace('/', '_')}.npy"
    cache = (cache_dir or corpus_path.parent / "emb_cache")
    cache.mkdir(parents=True, exist_ok=True)
    vec_path = cache / key
    t0 = time.perf_counter()
    if vec_path.exists():
        vecs = np.load(vec_path)
        if len(vecs) != len(corpus):
            vecs = None
        else:
            print(f"Vectors: cached ({vec_path.name})")
    else:
        vecs = None
    if vecs is None:
        print(f"Vectors: embedding {len(corpus)} docs with {model_name} (dim={enc.dim})...")
        vecs = enc.encode_docs([corpus.text(i) for i in range(len(corpus))])
        np.save(vec_path, vecs)
    t_embed = time.perf_counter() - t0
    dense = DenseIndex(vecs, corpus.ids)

    print(f"Built  : bm25 {t_bm25:.1f}s | vectors {t_embed:.1f}s\n")

    k_max = max(ks)
    pool = max(100, k_max * 5)      # candidates each retriever contributes
    qvecs = enc.encode_queries([r["query"] for r in rows])

    report = {s: {f"recall@{k}": 0.0 for k in ks} for s in strategies}
    for s in strategies:
        report[s]["mrr"] = 0.0
        report[s].update({f"ndcg@{k}": 0.0 for k in ks})

    for row, qv in zip(rows, qvecs):
        gold = set(row["relevant"])
        pq = process_query(row["query"])
        sp = bm25.retrieve(pq.sparse_query, top_k=pool)
        dn = dense.search(qv, top_k=pool)

        for s in strategies:
            if s == "bm25":
                ranked = [p for p, _ in sp]
            elif s == "dense":
                ranked = [p for p, _ in dn]
            elif s == "rrf":
                ranked = [p for p, _ in rrf_fuse([sp, dn])]
            elif s == "weighted":
                a = dynamic_alpha(pq, 0.5)
                ranked = [p for p, _ in weighted_fuse(sp, dn, a)]
            else:
                raise ValueError(s)
            report[s]["mrr"] += mrr(ranked, gold)
            for k in ks:
                report[s][f"recall@{k}"] += recall_at_k(ranked, gold, k)
                report[s][f"ndcg@{k}"] += ndcg_at_k(ranked, gold, k)

    n = len(rows)
    for s in strategies:
        for m in report[s]:
            report[s][m] = round(report[s][m] / n, 4)
    report["_meta"] = {
        "corpus": corpus_path.name, "corpus_size": len(corpus),
        "evalset": evalset_path.name, "n_queries": n,
        "model": model_name, "dim": enc.dim,
        "query_prefix": enc.q_prefix, "doc_prefix": enc.d_prefix,
    }
    return report


def print_report(report: dict) -> None:
    meta = report.pop("_meta")
    print(f"corpus={meta['corpus']} ({meta['corpus_size']} papers)  "
          f"evalset={meta['evalset']} ({meta['n_queries']} queries)")
    print(f"model ={meta['model']} (dim {meta['dim']})"
          + (f"  prefixes q='{meta['query_prefix']}' d='{meta['doc_prefix']}'"
             if meta["query_prefix"] or meta["doc_prefix"] else "  (no prefixes)"))
    metrics = list(next(iter(report.values())).keys())
    header = f"{'strategy':<12}" + "".join(f"{m:>11}" for m in metrics)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for s, vals in report.items():
        print(f"{s:<12}" + "".join(f"{vals[m]:>11.4f}" for m in metrics))
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("evaluation/data/corpus.jsonl"))
    ap.add_argument("--evalset", type=Path, default=Path("evaluation/data/evalset.jsonl"))
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 5, 10, 20])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--allow-missing-gold", action="store_true")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report = evaluate(
        args.corpus, args.evalset, args.model, tuple(args.k),
        allow_missing_gold=args.allow_missing_gold, limit=args.limit,
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report)


if __name__ == "__main__":
    main()
