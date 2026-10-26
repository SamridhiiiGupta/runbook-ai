import numpy as np
import pytest

from runbook_ai.chunker import Chunk
from runbook_ai.search import Retriever

WORDS = ["vpn", "password", "printer"]


def fake_embed(texts):
    """A tiny pretend embedder: counts how often each keyword appears."""
    return np.array([[t.lower().count(w) for w in WORDS] for t in texts], dtype=float)


CHUNKS = [
    Chunk("vpn.md", "Connect", "open the vpn client and connect to the vpn"),
    Chunk("pw.md", "Reset", "reset your password on the portal"),
    Chunk("printer.md", "Jam", "pull the paper out of the printer"),
]


def test_finds_most_relevant_chunk():
    r = Retriever(CHUNKS, embed_fn=fake_embed)
    top_chunk, score = r.search("my printer is broken", k=1)[0]
    assert top_chunk.source == "printer.md"
    assert score > 0


def test_returns_at_most_k_results():
    r = Retriever(CHUNKS, embed_fn=fake_embed)
    assert len(r.search("vpn", k=2)) == 2


def test_results_are_sorted_best_first():
    r = Retriever(CHUNKS, embed_fn=fake_embed)
    scores = [s for _, s in r.search("password reset", k=3)]
    assert scores == sorted(scores, reverse=True)


def test_empty_chunks_rejected():
    with pytest.raises(ValueError):
        Retriever([], embed_fn=fake_embed)