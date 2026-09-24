"""Build a retrieval eval set from citation contexts. No hand-written queries.

The idea
--------
When paper A cites paper B, the sentence A wrote around that citation is an
expert's natural-language description of B -- written without knowing our
system exists, in exactly the register a student uses when they describe the
paper they are looking for but cannot name.

    query  = that sentence, with the citation marker and any giveaway
             title/author tokens masked
    gold   = B's arXiv id

That gives thousands of labelled (query, relevant_doc) pairs for free, and
nobody on the team can accidentally write queries their own system happens to
answer -- which is what `evaluation/datasets/sample_eval.jsonl` did.

Why contexts come from the REFERENCES direction
-----------------------------------------------
Semantic Scholar only has citation contexts where it has parsed full text,
i.e. the open-access subset. Asking "who cites B" returns mostly closed-access
publisher records with empty contexts. Asking "what does A cite", where we
choose A from our own arXiv corpus, means the citing side is always open
access -- 38/40 refs came back with contexts in testing, versus 0/40 the
other way round.

Quality filtering is the whole game
-----------------------------------
Most citation sentences are useless as queries:

    "These approaches have proved successful in a number of domains
     including Machine Translation [18, 22] and Semantic Parsing [21]."

That describes a field, not a paper, and it cites three works at once. A
usable query describes ONE work specifically enough that a human could find
it. `_is_usable_context` encodes that; `--audit` prints what it kept and
threw away so the filter stays honest rather than becoming a way to hit a
target count.

Usage
-----
    python -m scripts.build_evalset --corpus evaluation/data/corpus.jsonl \
        --n-citing 400 --out evaluation/data/evalset.jsonl --audit
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

S2_REFS = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{}/references"

# Citation markers to strip: "[12]", "[12, 15]", "(Lewis et al., 2020)",
# "Lewis et al. (2020)". Leaving these in would let BM25 match on the
# bracket number, which measures nothing.
_BRACKET_CITE = re.compile(r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]")
_PAREN_CITE = re.compile(r"\(\s*[A-Z][A-Za-z\-']+(?:\s+et\s+al\.?)?(?:\s*(?:and|&)\s*[A-Z][A-Za-z\-']+)?\s*,?\s*(?:19|20)\d{2}[a-z]?\s*\)")
# Semicolon-separated citation piles: "(Chapelle et al., 2006; Lee et al.,
# 2013; Sajjadi et al., 2016; Laine & Aila, 2017)". _PAREN_CITE requires the
# paren to close right after the year, so it matches none of these and
# _count_citations scored the whole pile as one citation -- which is how
#   "A great number of early works (Chapelle et al., 2006; Lee et al., 2013;
#    Sajjadi et al., 2016; ...)"
# survived the multi_cite filter and became a "query" describing nothing.
_PAREN_PILE = re.compile(
    r"\(\s*[^)]*?(?:19|20)\d{2}[a-z]?\s*(?:;\s*[^)]*?(?:19|20)\d{2}[a-z]?\s*)+\)"
)
_NARRATIVE_CITE = re.compile(r"\b[A-Z][A-Za-z\-']+\s+et\s+al\.?\s*\(\s*(?:19|20)\d{2}[a-z]?\s*\)")
_YEAR_PAREN = re.compile(r"\(\s*(?:19|20)\d{2}[a-z]?\s*\)")

_WS = re.compile(r"\s+")
_STOP = {
    "the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "is", "are",
    "we", "our", "this", "that", "these", "those", "with", "by", "as", "it",
    "its", "their", "from", "at", "be", "been", "can", "which", "such", "has",
    "have", "was", "were", "also", "using", "used", "use", "more", "most",
}

MIN_CHARS, MAX_CHARS = 60, 400
MIN_CONTENT_WORDS = 6
MAX_CITES_IN_CONTEXT = 2   # a sentence citing 5 works describes none of them

# The citing paper narrating its OWN experiment is not a description of the
# cited work. "In Figure 3 we present reliability plots for ... DenseNet-161"
# says nothing about DenseNet that would help anyone find it, and no searcher
# types a sentence like it. These were roughly 40% of what the first version
# of this filter let through.
_SELF_NARRATION = re.compile(
    r"\b(?:we|our)\b"
    r"|\b(?:in|see)\s+(?:fig\.?|figure|tab\.?|table|sec\.?|section|appendix|§)\s*\d"
    r"|\bthis (?:paper|work|section|study)\b",
    re.I,
)

# A usable query says what the cited work IS or DOES. "use", "used" and
# "model" alone are too weak: they match "the backbone model is X", which
# after title masking leaves nothing findable behind.
_STRONG_DESCRIPTOR = re.compile(
    r"\b(?:propose|introduc|present(?:s|ed)?\s+a|develop|design(?:s|ed)?|"
    r"formulat|demonstrat|show(?:s|ed)?\s+that|argue|outperform|extend(?:s|ed)?|"
    r"generaliz|leverag|exploit|combin|replac|approximat|estimat|optimi[sz]|"
    r"achiev\w*\s+(?:state|sota|\d)|train(?:s|ed)?\s+(?:a|an|on)|"
    r"learn(?:s|ed)?\s+(?:a|an|to)|based\s+on|consists?\s+of|"
    r"framework\s+(?:for|that)|method\s+(?:for|that)|approach\s+(?:for|that|to))",
    re.I,
)


# S2 returns a window of text around the citation, often several sentences.
# Judging the whole window is wrong in both directions: a stray "we" in a
# neighbouring sentence discards a perfectly good description, and a good
# neighbouring sentence rescues a useless one. The sentence carrying the
# citation marker is the one that describes the cited work, so select it
# first and judge only that.
_CITE_MARK = re.compile(
    r"\[\s*\d+(?:\s*[,;-]\s*\d+)*\s*\]"
    r"|\(\s*[A-Z][A-Za-z\-']+(?:\s+et\s+al\.?)?[^)]{0,40}(?:19|20)\d{2}[a-z]?\s*\)"
    r"|\b[A-Z][A-Za-z\-']+\s+et\s+al\.?"
    r"|\(\s*\d{1,2}\s*\)"          # (7)-style numeric callouts
)
# Split on sentence enders, protecting common abbreviations that end in a dot.
_ABBREV = re.compile(r"\b(?:et al|e\.g|i\.e|cf|vs|Fig|Tab|Sec|Eq|Ref|approx|al)\.$", re.I)


# A sentence often ends "...quantization.(7) We show that..." or
# "...gain-shape quantization.[12] We show..." -- the character before the
# space is ')' or ']', not '.', so a naive (?<=[.!?])\s+ split never fires and
# the whole two-sentence window is judged as one. Allow a trailing citation
# marker between the full stop and the space.
_SENT_SPLIT = re.compile(r"(?<=[.!?])(?:\([^)]{0,15}\)|\[[^\]]{0,15}\])?\s+")


def _sentences(text: str) -> list[str]:
    parts, buf = [], []
    for tok in _SENT_SPLIT.split(text):
        buf.append(tok)
        if _ABBREV.search(tok.strip()):
            continue           # false ending ("et al." / "e.g.") -- keep going
        parts.append(" ".join(buf))
        buf = []
    if buf:
        parts.append(" ".join(buf))
    return [p.strip() for p in parts if p.strip()]


def _select_citing_sentence(raw: str) -> str:
    """The sentence containing the citation marker, else the longest one."""
    sents = _sentences(raw)
    if not sents:
        return raw
    marked = [s for s in sents if _CITE_MARK.search(s)]
    if marked:
        # If several, prefer the one with the most content around the marker.
        return max(marked, key=len)
    return max(sents, key=len)


def _clean(text: str) -> str:
    text = _PAREN_PILE.sub(" ", text)
    text = _NARRATIVE_CITE.sub(" ", text)
    text = _PAREN_CITE.sub(" ", text)
    text = _BRACKET_CITE.sub(" ", text)
    text = _YEAR_PAREN.sub(" ", text)
    text = text.replace("­", "").replace("ﬁ", "fi").replace("ﬂ", "fl")
    # PDF extraction hyphenates across line breaks: "perfor- mance"
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return _WS.sub(" ", text).strip()


def _count_citations(raw: str) -> int:
    n = len(_BRACKET_CITE.findall(raw)) + len(_PAREN_CITE.findall(raw))
    n += len(_NARRATIVE_CITE.findall(raw))
    # "[1, 2, 3]" is one bracket but three citations
    for m in _BRACKET_CITE.findall(raw):
        n += m.count(",")
    # "(A et al., 2006; B et al., 2013; C et al., 2016)" is one paren but
    # three citations -- semicolons count the works inside the pile.
    for m in _PAREN_PILE.findall(raw):
        n += m.count(";") + 1
    return max(n, 1)


# Mask a run of at least this many consecutive title words.
_MIN_TITLE_PHRASE = 3


def _mask_title_tokens(query: str, title: str) -> str:
    """Remove verbatim TITLE PHRASES, not individual title words.

    The first version masked any query word longer than four characters that
    also appeared in the title. That gutted exactly the queries it should
    have kept:

        gold  "Circadian Patterns of Wikipedia Editorial Activity"
        query "the broad studies on social aspects of and its communities
               of users makes it possible to..."

    "Wikipedia" was deleted as a title word, and with it the only token that
    made the paper findable. Same for TensorFlow and Hemingway. The result
    is a query that is both unanswerable and unlike anything a human types.

    The artifact actually worth removing is someone quoting the title, which
    would let BM25 win by string match without understanding anything. A
    searcher naturally using the topic word is not an artifact -- it is the
    normal case. So mask only runs of >= _MIN_TITLE_PHRASE consecutive title
    words, which catches quotation and leaves description alone.
    """
    t_words = [w.lower().strip(".,:;()") for w in title.split()]
    t_words = [w for w in t_words if w]
    q_tokens = query.split()
    q_bare = [w.lower().strip(".,:;()\"'") for w in q_tokens]

    title_ngrams = set()
    for n in range(_MIN_TITLE_PHRASE, min(len(t_words), 8) + 1):
        for i in range(len(t_words) - n + 1):
            title_ngrams.add(tuple(t_words[i:i + n]))

    drop = [False] * len(q_tokens)
    for n in range(min(8, len(q_bare)), _MIN_TITLE_PHRASE - 1, -1):
        for i in range(len(q_bare) - n + 1):
            if tuple(q_bare[i:i + n]) in title_ngrams:
                for j in range(i, i + n):
                    drop[j] = True
    kept = [w for w, d in zip(q_tokens, drop) if not d]
    return _WS.sub(" ", " ".join(kept)).strip()


def _is_usable_context(raw: str, cleaned: str, title: str) -> tuple[bool, str]:
    """Returns (keep, reason_if_dropped). Reasons are reported by --audit."""
    if len(cleaned) < MIN_CHARS:
        return False, "too_short"
    if len(cleaned) > MAX_CHARS:
        return False, "too_long"
    if _count_citations(raw) > MAX_CITES_IN_CONTEXT:
        return False, "multi_cite"

    words = [w.lower().strip(".,:;()") for w in cleaned.split()]
    content = [w for w in words if w not in _STOP and len(w) > 2]
    if len(content) < MIN_CONTENT_WORDS:
        return False, "thin_content"

    # Pure signposting sentences describe the section, not the work.
    low = cleaned.lower()
    if re.match(r"^(see|cf\.?|e\.g\.?|for (a )?(survey|review|overview|more))\b", low):
        return False, "signpost"
    if _SELF_NARRATION.search(cleaned):
        return False, "self_narration"
    if not _STRONG_DESCRIPTOR.search(cleaned):
        return False, "no_descriptor"

    # After masking title words, is anything substantive left?
    masked = _mask_title_tokens(cleaned, title)
    if len(masked.split()) < MIN_CONTENT_WORDS:
        return False, "empty_after_mask"
    return True, ""


# Raw S2 responses are cached to disk, one file per citing paper.
#
# The first full build took 700 minutes, almost all of it waiting on the
# API at ~2s per citing paper. Without a cache, every change to the quality
# filter -- and the filter needed four rounds of changes -- means paying
# that 11 hours again to re-download bytes we already had. With it, a
# filter change re-applies offline in seconds and the API is hit once.
S2_CACHE = Path("evaluation/data/s2_cache")


def fetch_references(
    client: httpx.Client, arxiv_id: str, limit: int = 100,
    use_cache: bool = True,
) -> list[dict]:
    cache_file = S2_CACHE / f"{arxiv_id.replace('/', '_')}.json"
    if use_cache and cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass          # corrupt cache entry: re-fetch below

    data = _fetch_references_uncached(client, arxiv_id, limit)
    if use_cache:
        S2_CACHE.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def _fetch_references_uncached(client: httpx.Client, arxiv_id: str, limit: int) -> list[dict]:
    for attempt in range(5):
        try:
            r = client.get(
                S2_REFS.format(arxiv_id),
                params={"fields": "contexts,title,externalIds", "limit": limit},
                timeout=45, follow_redirects=True,
            )
        except httpx.HTTPError:
            time.sleep(3 * (attempt + 1))
            continue
        if r.status_code == 200:
            return r.json().get("data", [])
        if r.status_code in (429, 504):        # rate limited / upstream timeout
            time.sleep(6 * (attempt + 1))
            continue
        return []                              # 404: not in S2, skip quietly
    return []


def build(
    corpus_path: Path, out_path: Path, n_citing: int,
    seed: int = 0, audit: bool = False, sleep_s: float = 1.1,
) -> dict:
    corpus = {}
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                corpus[rec["arxiv_id"]] = rec
    # arXiv ids carry versions in some sources ("2005.11401v3"); index bare too
    bare = {k.split("v")[0]: k for k in corpus}
    print(f"Corpus: {len(corpus)} papers")

    rng = random.Random(seed)
    # Prefer citing papers with substantial abstracts -- they tend to be full
    # papers with real related-work sections rather than 2-page notes.
    candidates = [k for k, v in corpus.items() if len(v.get("abstract", "")) > 600]
    rng.shuffle(candidates)

    rows: list[dict] = []
    drops: dict[str, int] = {}
    kept_examples: list[tuple[str, str]] = []
    dropped_examples: list[tuple[str, str]] = []
    n_api_hits = 0

    with httpx.Client(headers={"User-Agent": "arxiv-semantic-search/evalset"}) as client:
        for i, citing_id in enumerate(candidates):
            if len(rows) >= n_citing * 3 or i >= n_citing:
                break
            cached = (S2_CACHE / f"{citing_id.replace('/', '_')}.json").exists()
            refs = fetch_references(client, citing_id)
            if not cached:
                time.sleep(sleep_s)   # rate-limit only real API calls
            if not refs:
                continue
            n_api_hits += 1

            for ref in refs:
                cited = ref.get("citedPaper") or {}
                ext = cited.get("externalIds") or {}
                cid = ext.get("ArXiv")
                if not cid:
                    continue
                key = corpus.get(cid) and cid or bare.get(cid.split("v")[0])
                if not key:
                    continue                    # cited paper is outside our snapshot
                title = cited.get("title") or corpus[key]["title"]

                for raw_window in (ref.get("contexts") or []):
                    # narrow the S2 window to the sentence bearing the citation
                    raw = _select_citing_sentence(raw_window)
                    cleaned = _clean(raw)
                    ok, reason = _is_usable_context(raw, cleaned, title)
                    if not ok:
                        drops[reason] = drops.get(reason, 0) + 1
                        if audit and len(dropped_examples) < 12:
                            dropped_examples.append((reason, cleaned[:150] or raw[:150]))
                        continue
                    query = _mask_title_tokens(cleaned, title)
                    rows.append({
                        "query": query,
                        "relevant": [key],
                        "source": "citation_context",
                        "citing_paper": citing_id,
                        "cited_title": title,
                    })
                    if audit and len(kept_examples) < 12:
                        kept_examples.append((key, query[:150]))
                    break                       # one query per (citing, cited) pair

            if (i + 1) % 25 == 0:
                # flush: piped stdout is block-buffered, so without this an
                # 11-hour run shows no progress at all until it finishes.
                print(f"  {i+1:>4} citing papers scanned -> {len(rows)} queries",
                      flush=True)

    # One query per gold paper keeps the metric from being dominated by a
    # handful of heavily-cited works.
    by_gold: dict[str, dict] = {}
    for r in rows:
        by_gold.setdefault(r["relevant"][0], r)
    final = list(by_gold.values())
    rng.shuffle(final)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in final:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "queries": len(final),
        "before_dedup": len(rows),
        "citing_papers_used": n_api_hits,
        "dropped": dict(sorted(drops.items(), key=lambda x: -x[1])),
    }
    if audit:
        print("\n--- KEPT (sample) ---")
        for gid, q in kept_examples:
            print(f"  [{gid}] {q}")
        print("\n--- DROPPED (sample) ---")
        for reason, q in dropped_examples:
            print(f"  ({reason}) {q}")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=Path("evaluation/data/corpus.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("evaluation/data/evalset.jsonl"))
    ap.add_argument("--n-citing", type=int, default=400,
                    help="how many citing papers to pull references for")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--audit", action="store_true",
                    help="print kept/dropped examples so the filter stays honest")
    args = ap.parse_args()

    t0 = time.perf_counter()
    stats = build(args.corpus, args.out, args.n_citing, args.seed, args.audit)
    print(f"\nEval set: {stats['queries']} queries "
          f"({stats['before_dedup']} before one-per-gold dedup) "
          f"from {stats['citing_papers_used']} citing papers "
          f"in {(time.perf_counter()-t0)/60:.1f} min")
    print("Dropped contexts by reason:")
    for reason, n in stats["dropped"].items():
        print(f"  {reason:<18} {n}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
