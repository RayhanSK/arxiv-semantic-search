"""Tests for the offline seed corpus and arXiv rate-limit handling.

Covers the cold-start scenario from the field: live arXiv blocked/rate-
limited AND empty local corpus. The bundled seed corpus guarantees the
first search always has papers to rank, and the client now treats rate
limits as a first-class, explained condition instead of a hang.
"""
from __future__ import annotations

import json
import time

import httpx

import backend.app.services.arxiv_client as ac
from backend.app.services.arxiv_client import search_arxiv


# ------------------------------------------------------------ seed corpus
def test_seed_file_is_valid_jsonl():
    from backend.app.core.config import PROJECT_ROOT

    path = PROJECT_ROOT / "seed" / "arxiv_seed.jsonl"
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    assert len(recs) >= 40
    ids = [r["arxiv_id"] for r in recs]
    assert len(ids) == len(set(ids)), "seed ids must be unique"
    assert all(r["title"] and r["abstract"] and r["categories"] for r in recs)


def test_seed_ingestion_is_idempotent_and_searchable(pipeline):
    before = pipeline.bm25.size
    added = pipeline.seed_from_file()
    assert added >= 40
    assert pipeline.bm25.size == before + added
    # idempotent: re-seeding adds nothing (no-duplicate-indexing NFR)
    assert pipeline.seed_from_file() == 0

    out = pipeline.search("dense passage retrieval question answering",
                          fetch_arxiv=False)
    top = [r["arxiv_id"] for r in out["results"][:5]]
    assert "2004.04906" in top, "seeded DPR paper should rank for this query"


# ------------------------------------------------------- rate-limit client
def _reset_client_state(monkeypatch):
    monkeypatch.setattr(ac, "_last_fetch_ts", 0.0)
    monkeypatch.setattr(ac, "_cache", {})


def test_http_429_reported_as_rate_limited(monkeypatch):
    from backend.app.core.config import settings

    _reset_client_state(monkeypatch)
    monkeypatch.setattr(settings, "arxiv_min_interval_s", 0.0)
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    monkeypatch.setattr(
        ac.httpx, "get", lambda *a, **k: httpx.Response(429, request=req)
    )
    metas, err = search_arxiv("computer science")
    assert metas == []
    assert err and "rate-limited" in err and "429" in err


def test_rapid_fetches_are_throttled(monkeypatch):
    from backend.app.core.config import settings

    _reset_client_state(monkeypatch)
    monkeypatch.setattr(settings, "arxiv_min_interval_s", 3.0)
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return httpx.Response(500, request=req)

    monkeypatch.setattr(ac.httpx, "get", fake_get)
    search_arxiv("query one")            # real attempt (fails)
    metas, err = search_arxiv("query two")  # within the interval -> skipped
    assert calls["n"] == 1, "second rapid fetch must not hit the API"
    assert metas == [] and err and "throttled" in err


def test_identical_queries_served_from_cache(monkeypatch):
    from backend.app.core.config import settings

    from tests.test_arxiv_fetch import _ATOM_SAMPLE

    _reset_client_state(monkeypatch)
    monkeypatch.setattr(settings, "arxiv_min_interval_s", 0.0)
    req = httpx.Request("GET", "https://export.arxiv.org/api/query")
    calls = {"n": 0}

    def fake_get(*a, **k):
        calls["n"] += 1
        return httpx.Response(200, text=_ATOM_SAMPLE, request=req)

    monkeypatch.setattr(ac.httpx, "get", fake_get)
    t0 = time.perf_counter()
    m1, e1 = search_arxiv("dynamic alpha tuning")
    m2, e2 = search_arxiv("dynamic alpha tuning")
    assert calls["n"] == 1, "identical query within TTL must be cached"
    assert e1 is None and e2 is None
    assert [m.arxiv_id for m in m1] == [m.arxiv_id for m in m2] == ["2503.23013"]
    assert time.perf_counter() - t0 < 2
