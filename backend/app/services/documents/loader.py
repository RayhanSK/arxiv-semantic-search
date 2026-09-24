"""Lazy PDF loading and parsing.

PDFs are downloaded *only* when a paper survives hybrid retrieval +
reranking (the report's lazy-loading objective). Parsing uses PyMuPDF and
must never crash the pipeline: on any failure the paper falls back to
abstract-only mode and the error is recorded on the Paper row.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

import httpx

from backend.app.core.config import settings
from backend.app.db.models import Paper

logger = logging.getLogger(__name__)

try:
    import fitz  # PyMuPDF

    _FITZ_OK = True
except Exception:  # pragma: no cover
    _FITZ_OK = False


@dataclass
class ParsedDocument:
    arxiv_id: str
    # list of (heading, text) preserving document order; heading may be ""
    blocks: list[tuple[str, str]]
    n_pages: int = 0
    source: str = "pdf"  # "pdf" | "abstract"


def _pdf_path(arxiv_id: str) -> Path:
    return settings.pdf_dir / f"{arxiv_id.replace('/', '_')}.pdf"


def download_pdf(paper: Paper, timeout: int | None = None) -> Path | None:
    """Idempotent download; returns local path or None on failure."""
    path = _pdf_path(paper.arxiv_id)
    if path.exists() and path.stat().st_size > 1024:
        return path
    url = paper.pdf_url or f"https://arxiv.org/pdf/{paper.arxiv_id}"
    try:
        with httpx.Client(
            follow_redirects=True, timeout=timeout or settings.request_timeout_s
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            if not resp.content[:5].startswith(b"%PDF"):
                raise ValueError("response is not a PDF")
            path.write_bytes(resp.content)
        logger.info("Downloaded %s (%.1f KB)", paper.arxiv_id, len(resp.content) / 1024)
        return path
    except Exception as exc:
        logger.warning("PDF download failed for %s: %s", paper.arxiv_id, exc)
        paper.parse_error = f"download: {exc}"
        return None


# Section headings in arXiv/LaTeX papers: "1 Introduction", "3.2 Method",
# "II. RELATED WORK", "IV. EXPERIMENTS", or a bare canonical name.
_NUMBERED_HEADING = re.compile(
    r"^\s*(?:\d{1,2}(?:\.\d{1,2})*\.?|[IVXLC]{1,6}\.)\s+[A-Z][\w\-]*(?:\s+[\w\-]+){0,8}\s*$"
)
_BARE_HEADING = re.compile(
    r"^\s*(?:abstract|introduction|background|motivation|related\s+work|"
    r"prior\s+work|literature\s+review|method(?:s|ology)?|approach|"
    r"proposed\s+\w+|model|architecture|system\s+design|experiments?|"
    r"experimental\s+setup|evaluation|implementation|dataset[s]?|results?|"
    r"findings|analysis|ablation(?:\s+study)?|discussion|limitations?|"
    r"future\s+work|conclusions?(?:\s+and\s+future\s+work)?|summary|"
    r"references|bibliography|acknowledge?ments?|appendix|supplementary"
    r"(?:\s+material)?)\s*$",
    re.I,
)
# PyMuPDF span flag bit 4 (value 16) marks a bold face.
_BOLD_FLAG = 1 << 4

# Lines that are page furniture rather than content. Dropping them before
# heading detection matters twice over: the arXiv stamp printed down the
# margin of every submission is short and set in its own face, so it was
# being detected as a SECTION HEADING and every following paragraph was
# filed under "arXiv:2509.11056v2 [eess.SY] 1 Jun 2026"; and a bare page
# number on its own line gets glued to the next line of text, which is how
# a chunk came to read "1BERT is adopted because...".
_PAGE_FURNITURE = re.compile(
    r"^\s*(?:"
    r"arXiv:\d{4}\.\d{4,5}(?:v\d+)?.*"      # margin stamp
    r"|\d{1,4}"                              # bare page number
    r"|[ivxlcIVXLC]{1,6}"                    # roman page number
    r"|Page\s+\d+(?:\s+of\s+\d+)?"
    r"|\d{1,4}\s*/\s*\d{1,4}"
    r"|(?:Preprint|Under\s+review|Published\s+as).{0,60}"
    r")\s*$",
    re.I,
)


def _is_heading(text: str, size: float, body_size: float, bold: bool) -> bool:
    """Does this line start a new section?

    The original test was `size > body_size * 1.15`, which sounds reasonable
    and essentially never fires on real papers: LaTeX sets section headings
    BOLD at the same size as body text, or a few percent larger, not 15%.
    Measured on a corpus indexed with that rule, 388 of 429 chunks came back
    labelled "body" and exactly one was labelled "introduction" — so
    section-aware retrieval was section-aware in name only, and the
    references-list skip in chunker.py never triggered because reference
    lines were never labelled `references` either.

    Boldness and heading shape carry the signal that size does not.
    """
    if len(text) > 120 or len(text) < 3:
        return False
    if _NUMBERED_HEADING.match(text) or _BARE_HEADING.match(text):
        return True
    if bold and len(text) < 80 and text[0].isupper():
        return True
    return size > body_size * 1.08 and len(text) < 80


def _dehyphenate(text: str) -> str:
    """Rejoin words broken across a line wrap: "mod- els" -> "models".

    PDF extraction preserves the hyphen a typesetter inserted at a line
    break. Joining lines with a space turns it into "opti- mization", which
    then fails to match "optimization" in any retriever, is embedded as two
    junk tokens, and is read by a human as a typo in the answer. 328 of 429
    indexed chunks contained at least one.

    Only joins when the continuation is lowercase, so genuine compounds
    ("state-of-the-art", "GPU-based") survive.
    """
    return re.sub(r"([a-z])-\s+([a-z])", r"\1\2", text)


def parse_pdf(path: Path, arxiv_id: str) -> ParsedDocument | None:
    """Extract text blocks with heading detection by weight, shape and size."""
    if not _FITZ_OK:
        return None
    try:
        doc = fitz.open(path)
        sizes: list[float] = []
        raw_lines: list[tuple[float, bool, str]] = []  # (size, bold, text)
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = "".join(s.get("text", "") for s in spans).strip()
                    if not text or _PAGE_FURNITURE.match(text):
                        continue
                    size = max(s.get("size", 0.0) for s in spans)
                    bold = any(s.get("flags", 0) & _BOLD_FLAG for s in spans)
                    raw_lines.append((size, bold, text))
                    sizes.append(size)
        n_pages = doc.page_count
        doc.close()
        if not raw_lines:
            return None

        body_size = _median(sizes)
        blocks: list[tuple[str, str]] = []
        current_heading = ""
        buf: list[str] = []
        for size, bold, text in raw_lines:
            if _is_heading(text, size, body_size, bold):
                if buf:
                    blocks.append((current_heading, _dehyphenate(" ".join(buf))))
                    buf = []
                current_heading = text
            else:
                buf.append(text)
        if buf:
            blocks.append((current_heading, _dehyphenate(" ".join(buf))))
        return ParsedDocument(arxiv_id=arxiv_id, blocks=blocks, n_pages=n_pages)
    except Exception as exc:
        logger.warning("PDF parse failed for %s: %s", arxiv_id, exc)
        return None


def _median(values: list[float]) -> float:
    if not values:
        return 10.0
    s = sorted(values)
    return s[len(s) // 2]


def load_document(paper: Paper) -> ParsedDocument:
    """Full lazy pipeline for one paper: download -> parse -> fallback.

    Never raises; on failure returns an abstract-only document so QA can
    still answer from metadata (reliability NFR).
    """
    path = download_pdf(paper)
    if path is not None:
        parsed = parse_pdf(path, paper.arxiv_id)
        if parsed and parsed.blocks:
            paper.pdf_status = "parsed"
            paper.pdf_path = str(path)
            paper.parse_error = ""
            return parsed
        paper.parse_error = paper.parse_error or "parse: no extractable text"
    paper.pdf_status = "failed"
    logger.info("Falling back to abstract-only for %s", paper.arxiv_id)
    return ParsedDocument(
        arxiv_id=paper.arxiv_id,
        blocks=[("Abstract", paper.abstract or paper.title)],
        source="abstract",
    )
