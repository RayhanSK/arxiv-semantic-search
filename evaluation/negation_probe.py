"""Measure whether the retrieval stack can tell "X" from "not X".

Why this exists
---------------
A real query typed into the running system was:

    "hi how are you, i want papers that are related to computer networking
     but not related to llm or ai"

Every one of the ten results was an LLM paper. Three independent failures
compounded, and this script isolates the first and most fundamental one.

  1. The embedding model does not encode negation. "papers about X" and
     "papers not about X" are near-identical directions in vector space, so
     dense retrieval cannot separate them. That is what this script measures.

  2. Query expansion amplifies the excluded term. query_processor._EXPANSIONS
     maps "llm" -> "large language model", so the sparse query became
     "...not related to llm or ai large language model" -- actively searching
     for the thing the user ruled out.

  3. Nothing abstains. The cross-encoder scored the results 0.004 and 0.000,
     correctly saying "these do not match", and the UI displayed them anyway.

A cosine similarity near 1.0 between a query and its negation means the
distinction is invisible to the retriever, and no amount of fusion tuning or
reranking can recover information the encoder never represented. If the gap
is large, this thesis is wrong and we find that out cheaply.

Usage
-----
    python -m evaluation.negation_probe
    python -m evaluation.negation_probe --model BAAI/bge-base-en-v1.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Each pair differs ONLY by the negation. A retriever that understands the
# request should place these far apart; one that bags words will not.
PAIRS: list[tuple[str, str]] = [
    ("papers about computer networking that use machine learning",
     "papers about computer networking that do not use machine learning"),
    ("transformer models for time series forecasting",
     "time series forecasting models that avoid transformers"),
    ("image classification methods that rely on convolutional networks",
     "image classification methods that rely on no convolutional networks"),
    ("reinforcement learning approaches using deep neural networks",
     "reinforcement learning approaches without deep neural networks"),
    ("question answering systems built on large language models",
     "question answering systems built without large language models"),
    ("routing protocols evaluated with simulation",
     "routing protocols evaluated without simulation"),
    ("speech recognition trained on labelled data",
     "speech recognition trained with no labelled data"),
    ("graph algorithms that require a GPU",
     "graph algorithms that require no GPU"),
    ("recommender systems using collaborative filtering",
     "recommender systems that avoid collaborative filtering"),
    ("semantic segmentation with pretrained backbones",
     "semantic segmentation from scratch, no pretrained backbones"),
    ("optimisation methods that compute gradients",
     "gradient-free optimisation methods"),
    ("distributed training relying on parameter servers",
     "distributed training without parameter servers"),
    ("text generation evaluated by human annotators",
     "text generation evaluated without human annotators"),
    ("anomaly detection using labelled anomalies",
     "unsupervised anomaly detection with no labelled anomalies"),
    ("network intrusion detection using deep learning",
     "network intrusion detection not using deep learning"),
]

# A control: pairs that genuinely ARE about different things. If the encoder
# is working at all, these must score lower than the negation pairs -- so a
# negation score at or above the control means negation is invisible.
CONTROLS: list[tuple[str, str]] = [
    ("papers about computer networking", "papers about protein folding"),
    ("transformer models for time series", "database index structures"),
    ("image classification with CNNs", "formal verification of compilers"),
    ("reinforcement learning for robotics", "typography and font rendering"),
    ("question answering over documents", "quantum error correction codes"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    from evaluation.harness import Encoder

    enc = Encoder(args.model)
    print(f"model: {args.model} (dim {enc.dim})")
    print("A pair that differs only by negation SHOULD score low.\n")

    def sims(pairs: list[tuple[str, str]]) -> np.ndarray:
        a = enc.encode_queries([p[0] for p in pairs])
        b = enc.encode_queries([p[1] for p in pairs])
        return np.sum(a * b, axis=1)      # both are L2-normalised

    neg = sims(PAIRS)
    ctl = sims(CONTROLS)

    print(f"{'cos':>7}  query pair (differs only by negation)")
    print("-" * 78)
    for s, (x, _) in sorted(zip(neg, PAIRS), key=lambda t: -t[0]):
        print(f"{s:>7.3f}  {x[:66]}")

    print(f"\n{'cos':>7}  CONTROL pair (genuinely unrelated topics)")
    print("-" * 78)
    for s, (x, y) in sorted(zip(ctl, CONTROLS), key=lambda t: -t[0]):
        print(f"{s:>7.3f}  {x[:30]:<32} vs {y[:30]}")

    print("\n" + "=" * 78)
    print(f"negation pairs : mean {neg.mean():.3f}   min {neg.min():.3f}   max {neg.max():.3f}")
    print(f"control pairs  : mean {ctl.mean():.3f}   min {ctl.min():.3f}   max {ctl.max():.3f}")
    print(f"separation     : {neg.mean() - ctl.mean():+.3f}")
    print()
    if neg.mean() > 0.80:
        print("VERDICT: negation is invisible to this encoder. A query and its")
        print("opposite are near-duplicates in vector space, so no fusion weight,")
        print("threshold or reranker can recover a distinction the vectors never")
        print("carried. This has to be handled before retrieval, by parsing the")
        print("constraint out of the query -- not after it, by ranking better.")
    else:
        print("VERDICT: this encoder separates negated pairs better than assumed.")
        print("Re-examine the premise before building constraint parsing on it.")


if __name__ == "__main__":
    main()
