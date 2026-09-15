"""Regression tests for the hanging-arXiv-fetch failure.

Reproduces: live search hung ~80s inside the arxiv package's untimed HTTP
call, returned nothing, and the empty local corpus produced a blank screen
with no explanation. The fetch must be hard-bounded and failures must
surface as a user-visible notice.
"""
from __future__ import annotations

import time

import backend.app.services.pipeline as pl
from backend.app.services.arxiv_client import parse_atom_feed, search_arxiv

_ATOM_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2503.23013v2</id>
    <title>DAT: Dynamic  Alpha
      Tuning</title>
    <summary>We propose   dynamic alpha tuning.</summary>
    <published>2025-03-29T12:00:00Z</published>
    <author><name>A. Researcher</name></author>
    <author><name>B. Scholar</name></author>
    <link href="http://arxiv.org/abs/2503.23013v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2503.23013v2" rel="related" type="application/pdf"/>
    <category term="cs.IR"/><category term="cs.CL"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/9999.99999v1</id>
    <title></title>
  </entry>
</feed>"""


def test_atom_feed_parsing():
    metas = parse_atom_feed(_ATOM_SAMPLE)
    assert len(metas) == 1  # titleless entry dropped
    m = metas[0]
    assert m.arxiv_id == "2503.23013"
    assert m.title == "DAT: Dynamic Alpha Tuning"        # whitespace squashed
    assert m.abstract == "We propose dynamic alpha tuning."
    assert m.authors == "A. Researcher; B. Scholar"
    assert m.categories == "cs.IR,cs.CL"
    assert m.published == "2025-03-29"
    assert m.pdf_url.endswith("/pdf/2503.23013v2")


def test_unreachable_arxiv_api_fails_fast_with_reason(monkeypatch):
    from backend.app.core.config import settings

    monkeypatch.setattr(settings, "arxiv_api_url", "http://127.0.0.1:9/api/query")
    monkeypatch.setattr(settings, "arxiv_timeout_s", 2.0)
    t0 = time.perf_counter()
    metas, err = search_arxiv("computer science")
    elapsed = time.perf_counter() - t0
    assert metas == []
    assert err and "arXiv API" in err
    assert elapsed < 5, "fetch failure must be bounded, never a hang"


def test_fetch_failure_surfaces_notice_and_keeps_local_results(pipeline, monkeypatch):
    monkeypatch.setattr(
        pl, "search_arxiv",
        lambda q, max_results=None: ([], "arXiv API timed out after 8s"),
    )
    out = pipeline.search("hybrid retrieval reranking", fetch_arxiv=True)
    assert out["notice"] and "timed out" in out["notice"]
    assert "local corpus" in out["notice"]
    assert out["results"], "local corpus results must still be returned"
    assert out["timings_ms"]["arxiv_fetch"] < 1000


def test_successful_local_search_has_no_notice(pipeline):
    out = pipeline.search("hybrid retrieval", fetch_arxiv=False)
    assert out["notice"] is None
    assert out["results"]
