"""Prompt templates.

Contract shared by every prompt:
* answers are grounded ONLY in the numbered context excerpts;
* every factual claim carries an inline bracketed citation, e.g. [2];
* missing information is stated explicitly, never guessed.

The templates also prescribe an answer *structure* (direct answer first,
then explanation) so that stronger generative backends produce clear,
well-organized responses rather than a wall of text. ``_parse_prompt`` in
``llm/generator.py`` depends on the ``[n] label\\ncontext`` /
``Question:`` layout produced by ``format_contexts`` — keep them in sync.
"""

SYSTEM_QA = (
    "You are an expert research assistant answering questions about scientific "
    "papers.\n"
    "Rules you must always follow:\n"
    "1. GROUNDING — use ONLY the numbered context excerpts provided. Never use "
    "outside knowledge, and never invent results, numbers, or method names.\n"
    "2. CITATIONS — every factual claim must end with the bracketed number(s) "
    "of the excerpt(s) supporting it, e.g. [2] or [1][3]. A sentence without a "
    "citation must contain no factual claim.\n"
    "3. HONESTY — if the excerpts do not contain the answer (or only part of "
    "it), say exactly which part is missing with: 'The provided context does "
    "not contain ...'. Do not fill gaps by guessing.\n"
    "4. STRUCTURE — begin with a direct 1–2 sentence answer, then explain the "
    "supporting details in short paragraphs. Use a brief bulleted list only "
    "when comparing multiple items. Define technical terms in plain language "
    "the first time they appear.\n"
    "5. STYLE — be precise and concise; no filler, no preamble like 'Based on "
    "the context', no repetition."
)

SINGLE_DOC_TEMPLATE = """Context excerpts from the paper "{title}" (arXiv:{arxiv_id}):

{contexts}

{history}Question: {question}

Answer using only the excerpts above. Start with a direct answer, then explain \
the details, citing the excerpt number(s) after every claim, e.g. [2]. If the \
excerpts only partially answer the question, answer the supported part and \
state what is missing."""

MAP_TEMPLATE = """Context excerpts from the paper "{title}" (arXiv:{arxiv_id}):

{contexts}

Question: {question}

List every fact in these excerpts that helps answer the question, as complete, \
self-contained sentences (keep concrete numbers, method names, and findings; \
add no interpretation of your own). If nothing in the excerpts is relevant, \
reply exactly: NO RELEVANT INFORMATION."""

REDUCE_TEMPLATE = """You are synthesizing one answer from evidence extracted \
from several papers. Each numbered block below is the evidence from one paper.

Per-paper evidence:

{contexts}

Question: {question}

Write one coherent, well-structured answer:
- Start with a direct 1–2 sentence synthesis that answers the question.
- Then compare the papers: where they agree, where they differ or conflict, \
and what each uniquely contributes.
- Cite the paper number(s) after every claim, e.g. [1] or [1][3]; do not cite \
papers for claims they do not support.
- Do not repeat the same point twice, and do not introduce information absent \
from the evidence."""


def format_contexts(items: list[dict], text_key: str = "text") -> str:
    """items: [{'label': 'title/section', text_key: ...}] -> numbered blocks."""
    lines = []
    for i, it in enumerate(items, start=1):
        label = it.get("label", "")
        lines.append(f"[{i}] {label}\n{it[text_key]}")
    return "\n\n".join(lines)
