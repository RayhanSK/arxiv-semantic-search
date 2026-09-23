"""Bulk-harvest a FROZEN arXiv metadata snapshot via OAI-PMH.

Why this exists
---------------
`pipeline.search()` calls the live arXiv API and ingests whatever comes back,
mid-query. That is fine for a demo and fatal for measurement: the corpus is
different on every run, so no retrieval number is reproducible and no two
strategies are ever compared over the same index.

This script produces a snapshot file that does not move. Every evaluation
result is reported against a named snapshot, and re-running the evaluation a
month later against the same snapshot gives the same number.

Why OAI-PMH rather than the search API
--------------------------------------
The public search API returns <=100 records per request and asks for ~3s
between calls; 50k papers would be ~25 minutes of pure waiting plus a fragile
offset-pagination scheme that arXiv caps around 30k results per query.
OAI-PMH is arXiv's actual bulk interface: ~1000 records per page, a
resumption-token cursor that cannot skip or duplicate records, and a
documented 503 + Retry-After backpressure protocol.

Why the corpus is harvested in YEAR WINDOWS
-------------------------------------------
The first version of this script pulled one contiguous range and produced a
corpus that was 88% 2024 papers, with ~10 papers per year for 2017-2020. That
corpus is useless for citation-derived ground truth: references overwhelmingly
point at work from 2017-2022, so almost no cited paper was in the index and
the eval set came out nearly empty.

Sweeping a quota per year instead gives an era-balanced pool, which is both
what a real search corpus looks like and what makes citation overlap possible.
Note that OAI `from`/`until` filter on the record's last-modified datestamp,
not its submission date, so a window bleeds slightly across years -- the
per-year quota is approximate on purpose.

Usage
-----
    python -m scripts.harvest --years 2017-2025 --per-year 6000
    python -m scripts.harvest --from 2024-01-01 --until 2025-06-30 --target 50000

Resumable: the cursor is checkpointed after every page, so a killed run
continues rather than restarting.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

# Windows consoles are cp1252 and paper titles are full of ligatures, accents
# and maths. Without this the harvester dies on a "ﬁ" 4000 papers in.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

OAI_URL = "http://export.arxiv.org/oai2"
NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "arxiv": "http://arxiv.org/OAI/arXiv/",
}

DEFAULT_OUT = Path("evaluation/data/corpus.jsonl")
DEFAULT_STATE = Path("evaluation/data/harvest_state.json")

# arXiv returns 503 with Retry-After when it wants you to slow down. Honour it
# rather than hammering: a banned IP costs more than a slow harvest.
MAX_RETRIES = 8


def _text(node, path: str) -> str:
    el = node.find(path, NS)
    return (el.text or "").strip() if el is not None and el.text else ""


def _parse_record(rec) -> dict | None:
    """One OAI <record> -> the flat dict the ingest pipeline expects."""
    meta = rec.find("oai:metadata/arxiv:arXiv", NS)
    if meta is None:
        return None  # deleted record (header-only, status="deleted")

    arxiv_id = _text(meta, "arxiv:id")
    title = " ".join(_text(meta, "arxiv:title").split())
    abstract = " ".join(_text(meta, "arxiv:abstract").split())
    if not arxiv_id or not title:
        return None

    authors = []
    for a in meta.findall("arxiv:authors/arxiv:author", NS):
        name = f"{_text(a, 'arxiv:forenames')} {_text(a, 'arxiv:keyname')}".strip()
        if name:
            authors.append(name)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "abstract": abstract,
        "authors": ", ".join(authors),
        "categories": _text(meta, "arxiv:categories"),
        "published": _text(meta, "arxiv:created"),
        "updated": _text(meta, "arxiv:updated"),
        "doi": _text(meta, "arxiv:doi"),
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
    }


def _fetch(client: httpx.Client, params: dict) -> str:
    """GET one OAI page, honouring 503/Retry-After backpressure."""
    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(OAI_URL, params=params, timeout=90, follow_redirects=True)
        except httpx.HTTPError as exc:
            wait = min(60, 5 * 2**attempt)
            print(f"  ! {type(exc).__name__}: {exc}; retry in {wait}s")
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return r.text
        if r.status_code == 503:
            wait = int(r.headers.get("Retry-After", 20))
            print(f"  . arXiv asked us to wait {wait}s (503)")
            time.sleep(wait)
            continue
        raise RuntimeError(f"OAI-PMH HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError(f"giving up after {MAX_RETRIES} retries")


def harvest(
    date_from: str,
    date_until: str,
    target: int,
    out_path: Path,
    state_path: Path,
    oai_set: str = "cs",
    resume: bool = True,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    token: str | None = None
    if resume and state_path.exists() and out_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        token = state.get("token")
        with out_path.open(encoding="utf-8") as fh:
            seen = {json.loads(line)["arxiv_id"] for line in fh if line.strip()}
        print(f"Resuming: {len(seen)} papers already on disk, token={bool(token)}")
    else:
        out_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)

    mode = "a" if seen else "w"
    page = 0
    with httpx.Client(headers={"User-Agent": "arxiv-semantic-search/eval-harvest"}) as client, \
            out_path.open(mode, encoding="utf-8") as fh:
        while len(seen) < target:
            # A resumption token is exclusive of every other argument — arXiv
            # rejects the request if you re-send set/from/until alongside it.
            params = (
                {"verb": "ListRecords", "resumptionToken": token}
                if token
                else {
                    "verb": "ListRecords",
                    "set": oai_set,
                    "metadataPrefix": "arXiv",
                    "from": date_from,
                    "until": date_until,
                }
            )
            xml = _fetch(client, params)
            root = ET.fromstring(xml)

            err = root.find("oai:error", NS)
            if err is not None:
                print(f"OAI error [{err.get('code')}]: {err.text}")
                break

            page += 1
            added = 0
            for rec in root.findall("oai:ListRecords/oai:record", NS):
                paper = _parse_record(rec)
                if paper is None or paper["arxiv_id"] in seen:
                    continue
                seen.add(paper["arxiv_id"])
                fh.write(json.dumps(paper, ensure_ascii=False) + "\n")
                added += 1
                if len(seen) >= target:
                    break
            fh.flush()

            tok_el = root.find("oai:ListRecords/oai:resumptionToken", NS)
            token = (tok_el.text or "").strip() if tok_el is not None else ""
            complete = root.find("oai:ListRecords/oai:resumptionToken", NS)
            total = complete.get("completeListSize") if complete is not None else "?"
            print(f"  page {page:>4}: +{added:<5} total={len(seen):<7} of {total} available")

            # Checkpoint after every page: a killed run resumes here.
            state_path.write_text(
                json.dumps({"token": token, "count": len(seen),
                            "from": date_from, "until": date_until, "set": oai_set}),
                encoding="utf-8",
            )
            if not token:
                print("  (no resumption token -- end of the date range)")
                break
            time.sleep(1.0)  # be a good citizen even when not asked to

    return len(seen)


def harvest_years(
    year_from: int, year_to: int, per_year: int,
    out_path: Path, state_path: Path, oai_set: str = "cs",
) -> int:
    """Sweep one quota per calendar year into a single snapshot file.

    Each year keeps its own cursor file so a crash mid-sweep resumes at the
    right year rather than re-downloading everything.
    """
    total = 0
    for year in range(year_from, year_to + 1):
        yr_out = out_path.with_name(f"{out_path.stem}.{year}.jsonl")
        yr_state = state_path.with_name(f"{state_path.stem}.{year}.json")
        done = sum(1 for _ in yr_out.open(encoding="utf-8")) if yr_out.exists() else 0
        if done >= per_year:
            print(f"[{year}] already have {done}, skipping")
            total += done
            continue
        print(f"[{year}] harvesting up to {per_year}...")
        n = harvest(
            f"{year}-01-01", f"{year}-12-31", per_year,
            yr_out, yr_state, oai_set, resume=True,
        )
        total += n

    # Merge the per-year files into the single snapshot, deduping by id.
    seen: set[str] = set()
    with out_path.open("w", encoding="utf-8") as fh:
        for year in range(year_from, year_to + 1):
            yr_out = out_path.with_name(f"{out_path.stem}.{year}.jsonl")
            if not yr_out.exists():
                continue
            for line in yr_out.open(encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["arxiv_id"] in seen:
                    continue
                seen.add(rec["arxiv_id"])
                fh.write(line)
    print(f"\nMerged {len(seen)} unique papers -> {out_path}")
    return len(seen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="date_from", default="2024-01-01")
    ap.add_argument("--until", dest="date_until", default="2025-06-30")
    ap.add_argument("--target", type=int, default=50_000)
    ap.add_argument("--years", help="year sweep, e.g. 2017-2025 (overrides --from/--until)")
    ap.add_argument("--per-year", type=int, default=6000)
    ap.add_argument("--set", dest="oai_set", default="cs")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--state", type=Path, default=DEFAULT_STATE)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.years:
        y0, y1 = (int(x) for x in args.years.split("-"))
        print(f"Harvesting arXiv set '{args.oai_set}' {y0}-{y1}, "
              f"~{args.per_year}/year -> {args.out}\n")
        n = harvest_years(y0, y1, args.per_year, args.out, args.state, args.oai_set)
    else:
        print(f"Harvesting arXiv set '{args.oai_set}' {args.date_from} -> {args.date_until}")
        print(f"Target {args.target} papers -> {args.out}\n")
        n = harvest(
            args.date_from, args.date_until, args.target,
            args.out, args.state, args.oai_set, resume=not args.no_resume,
        )
    mins = (time.perf_counter() - t0) / 60
    print(f"\nDone: {n} papers in {mins:.1f} min -> {args.out}")
    print("This file is the frozen snapshot. Quote its name with every eval result.")


if __name__ == "__main__":
    main()
