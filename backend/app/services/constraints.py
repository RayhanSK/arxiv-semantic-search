"""Parse hard constraints out of a query before retrieval runs.

The problem this exists to solve
--------------------------------
A user typed, into the running UI:

    "i want papers that are related to computer networking but not
     related to llm or ai"

Every one of the ten results was an LLM paper. Measured over 15 query pairs
that differ only by a negation, against 5 unrelated pairs as control:

    all-MiniLM-L6-v2   negated pairs 0.890 mean   unrelated pairs 0.066
    bge-base-en-v1.5   negated pairs 0.884 mean   unrelated pairs 0.488

A query and its literal opposite sit at cosine ~0.89. "question answering
built ON large language models" versus "built WITHOUT large language
models" scores 0.959. The distinction is simply not present in the vectors,
and a stronger encoder does not add it -- BGE scores negated pairs the same
while inflating everything else.

That rules out every fix the retrieval stack is shaped for. No fusion
weight, rerank threshold or corrective retry can recover information the
embedding never carried. So the exclusion has to be taken out of the query
and applied as a filter, which is what this module does.

It made things actively worse, too
----------------------------------
`query_processor._EXPANSIONS` maps "llm" -> "large language model". It has
no idea the term sits under a negation, so the sparse query became

    "...but not related to llm or ai large language model"

-- the system searching harder for precisely what the user excluded.
`parse_constraints` returns the negated spans so expansion can skip them.

Deliberately rule-based
-----------------------
An LLM call would parse these more flexibly, but it would add a network
round trip to every search, cost money, and be nondeterministic in a demo.
Negation cues in search queries are a small, closed set. When a cue is not
recognised the query degrades to exactly today's behaviour, so the failure
mode is "no worse than before" rather than "confidently wrong".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Cues that introduce an exclusion, and how far the exclusion reaches.
# Scope runs from the cue to the next clause boundary, because "not about
# X, and I also want Y" must not exclude Y.
#
# Every cue must contain an explicit negation token. An earlier version
# wrote the contracted forms as `(?:are|is)\s*n?o?t?`, in which `n?o?t?`
# matches the empty string -- so "papers that ARE RELATED TO computer
# networking" parsed as an exclusion and the topic the user asked for was
# stripped out of their own query. Optional negation is not negation.
_NEG_CUES = (
    r"but\s+not\s+(?:related\s+to|about|on|involving|using)?",
    r"not\s+(?:related\s+to|about|involving|using)",
    r"(?:that|which)\s+(?:are|is|were|was|do|does|did|can|could|will|would)\s+not\s+"
    r"(?:related\s+to|about|involving|using|require|requiring|need|use|rely\s+on)?",
    r"(?:that|which)\s+(?:aren|isn|weren|wasn|don|doesn|didn)'?t\s+"
    r"(?:related\s+to|about|involving|using)?",
    r"nothing\s+(?:to\s+do\s+with|about)",
    r"unrelated\s+to",
    r"excluding",
    r"except\s+(?:for\s+)?",
    r"without",
    r"other\s+than",
    r"no\s+(?!more\b)",
    r"non[-\s]",
)
_NEG_RE = re.compile(r"\b(?:" + "|".join(_NEG_CUES) + r")\s+", re.I)

# An exclusion ends at a clause boundary, not at the end of the string.
_SCOPE_END = re.compile(
    r"[.;]|,\s*(?:and|but|while|whereas)\b|\band\s+(?:i|we)\s+(?:also\s+)?want\b"
    r"|\bbut\s+(?:i|we)\b", re.I
)

# Terms joined inside one exclusion: "not about llm or ai", "no CNNs and RNNs"
_SPLIT_TERMS = re.compile(r"\s*(?:,|/|\bor\b|\band\b|\bnor\b)\s*", re.I)

_STOP_IN_TERM = {
    "the", "a", "an", "of", "for", "to", "any", "anything", "papers", "paper",
    "work", "works", "research", "stuff", "things", "topic", "topics", "it",
    "them", "that", "this", "those", "these", "related", "about",
}

# Expanding an excluded term matters more than expanding an included one:
# a user who says "not ai" means the concept, and papers say "artificial
# intelligence", "LLM", "deep learning" instead. Missing a surface form
# means the thing they excluded comes straight back.
_EXCLUDE_SYNONYMS: dict[str, list[str]] = {
    "ai": ["artificial intelligence", "machine learning", "deep learning",
           "neural network", "llm", "large language model", "transformer",
           "gpt", "chatgpt", "foundation model"],
    "llm": ["large language model", "language model", "gpt", "chatgpt",
            "instruction tuning", "foundation model"],
    "llms": ["large language model", "language model", "gpt", "chatgpt"],
    "ml": ["machine learning", "deep learning", "neural network"],
    "machine learning": ["deep learning", "neural network", "supervised learning"],
    "deep learning": ["neural network", "cnn", "transformer"],
    "neural networks": ["neural network", "deep learning", "cnn", "rnn"],
    "transformers": ["transformer", "attention", "self-attention"],
    "attention": ["self-attention", "transformer"],
    "cnn": ["convolutional neural network", "convolution"],
    "rnn": ["recurrent neural network", "lstm", "gru"],
    "gpu": ["cuda", "graphics processing unit"],
    "supervised": ["labelled data", "labeled data", "supervision"],
    "reinforcement learning": ["rl", "policy gradient", "q-learning"],
    "quantum": ["quantum computing", "qubit"],
    "blockchain": ["distributed ledger", "cryptocurrency"],
}

# arXiv primary categories that ARE the excluded topic. Dropping on category
# is far more reliable than string matching an abstract: an AI paper need
# never contain the word "AI".
_EXCLUDE_CATEGORIES: dict[str, list[str]] = {
    "ai": ["cs.AI", "cs.CL", "cs.LG", "cs.NE", "cs.CV", "stat.ML"],
    "llm": ["cs.CL"],
    "llms": ["cs.CL"],
    "ml": ["cs.LG", "stat.ML", "cs.NE"],
    "machine learning": ["cs.LG", "stat.ML", "cs.NE"],
    "deep learning": ["cs.LG", "cs.NE", "stat.ML"],
    "nlp": ["cs.CL"],
    "computer vision": ["cs.CV"],
    "vision": ["cs.CV"],
    "robotics": ["cs.RO"],
    "cryptography": ["cs.CR"],
    "security": ["cs.CR"],
    "networking": ["cs.NI"],
    "databases": ["cs.DB"],
    "theory": ["cs.CC", "cs.DS"],
}


@dataclass
class QueryConstraints:
    """What the user asked for, split from what they ruled out."""

    positive_text: str                                   # drives retrieval
    exclude_terms: list[str] = field(default_factory=list)
    exclude_categories: list[str] = field(default_factory=list)
    negated_spans: list[str] = field(default_factory=list)  # for expansion to skip

    @property
    def has_exclusions(self) -> bool:
        return bool(self.exclude_terms or self.exclude_categories)

    def describe(self) -> str:
        """Human-readable, for the UI. Silent filtering is its own bug."""
        if not self.has_exclusions:
            return ""
        bits = []
        if self.exclude_terms:
            bits.append("terms: " + ", ".join(sorted(set(self.exclude_terms))[:6]))
        if self.exclude_categories:
            bits.append("categories: " + ", ".join(sorted(set(self.exclude_categories))))
        return "excluding " + "; ".join(bits)


def _clean_term(raw: str) -> str:
    t = raw.strip().strip(".,;:!?\"'()").lower()
    words = [w for w in t.split() if w not in _STOP_IN_TERM]
    return " ".join(words).strip()


def parse_constraints(query: str) -> QueryConstraints:
    """Split a query into its positive request and its exclusions."""
    exclude_terms: list[str] = []
    exclude_cats: list[str] = []
    negated_spans: list[str] = []
    groups: list[tuple[str, list[str], list[str]]] = []
    positive = query

    for m in _NEG_RE.finditer(query):
        start = m.end()
        rest = query[start:]
        end_m = _SCOPE_END.search(rest)
        span = rest[: end_m.start()] if end_m else rest
        span = span.strip()
        if not span:
            continue
        negated_spans.append(span)

        for raw_term in _SPLIT_TERMS.split(span):
            term = _clean_term(raw_term)
            if not term or len(term) < 2:
                continue
            # Remember which negated root each synonym came from, so a
            # collision with the positive query can retract the whole
            # inference rather than one surface form of it.
            groups.append((term, _EXCLUDE_SYNONYMS.get(term, []),
                           _EXCLUDE_CATEGORIES.get(term, [])))

        # Remove the whole negated clause from the text used for retrieval,
        # so the encoder never sees the excluded topic at all. This is the
        # actual fix: the embedding cannot represent "not X", so "not X"
        # must not reach it.
        positive = positive.replace(query[m.start(): start] + span, " ")

    positive = re.sub(r"\s+", " ", positive).strip(" ,;.")
    if not positive:                    # the query was nothing but exclusions
        positive = query

    # Never exclude something the user also ASKED FOR.
    #
    # "computer networking concepts for running large language models in a
    # data centre, not related to AI" expands "ai" to a synonym list that
    # includes "large language model" -- a phrase from the positive half of
    # the same sentence. Without this guard the filter rejected
    # "Alibaba HPN: A Data Center Network for Large Language Model Training",
    # the exact paper being asked for, because it "mentions 'large language
    # model'". The category rule banned it a second time via cs.LG.
    #
    # Retracting only the one colliding synonym is not enough: "llm" would
    # still ban the same paper. A collision means our EXPANSION of what the
    # user meant contradicts what they typed, so the whole inference from
    # that root is dropped -- synonyms and categories both. The literal word
    # they negated is kept only if it does not itself appear in the request.
    pos_low = " " + positive.lower() + " "

    def _in_positive(term: str) -> bool:
        # allow a plural: "large language models" in the request must match
        # the excluded term "large language model".
        return bool(re.search(
            r"(?<![a-z0-9])" + re.escape(term) + r"(?:e?s)?(?![a-z0-9])", pos_low))

    for root, syns, cats in groups:
        collides = _in_positive(root) or any(_in_positive(x) for x in syns)
        if collides:
            continue                    # the user asked for this; do not ban it
        exclude_terms.append(root)
        exclude_terms.extend(syns)
        exclude_cats.extend(cats)

    return QueryConstraints(
        positive_text=positive,
        exclude_terms=exclude_terms,
        exclude_categories=list(dict.fromkeys(exclude_cats)),
        negated_spans=negated_spans,
    )


def violates(paper: dict, c: QueryConstraints) -> str | None:
    """Return the reason this paper is excluded, or None if it is allowed.

    Returning the reason rather than a bool means the UI can say WHY a
    result was withheld, which keeps the filter auditable instead of
    mysterious.
    """
    if not c.has_exclusions:
        return None

    cats = {x.strip() for x in (paper.get("categories") or "").replace(",", " ").split()}
    for bad in c.exclude_categories:
        if bad in cats:
            return f"category {bad}"

    haystack = f"{paper.get('title','')} {paper.get('abstract','')}".lower()
    for term in c.exclude_terms:
        # Word-boundary match so "ai" does not fire on "chain" or "available".
        if re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", haystack):
            return f"mentions '{term}'"
    return None
