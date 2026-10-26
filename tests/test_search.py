import numpy as np

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


def test_empty_retriever_returns_nothing():
    assert Retriever(embed_fn=fake_embed).search("vpn") == []


def test_add_chunks_updates_the_library():
    r = Retriever(embed_fn=fake_embed)
    r.add_chunks(CHUNKS)
    assert r.sources() == {"vpn.md": 1, "pw.md": 1, "printer.md": 1}


def test_remove_source():
    r = Retriever(CHUNKS, embed_fn=fake_embed)
    assert r.remove_source("vpn.md") == 1
    assert "vpn.md" not in r.sources()
    assert r.search("printer", k=1)[0][0].source == "printer.md"


def test_removing_everything_leaves_an_empty_index():
    r = Retriever(CHUNKS, embed_fn=fake_embed)
    for name in list(r.sources()):
        r.remove_source(name)
    assert r.search("vpn") == []


def test_replace_source_swaps_old_for_new():
    r = Retriever(CHUNKS, embed_fn=fake_embed)
    r.replace_source("vpn.md", [Chunk("vpn.md", "New", "vpn reconnect steps")])
    assert len(r.chunks) == 3
    assert all("open the vpn client" not in c.text for c in r.chunks)


def test_save_and_load_roundtrip(tmp_path):
    r = Retriever(CHUNKS, embed_fn=fake_embed)
    r.save(tmp_path)
    loaded = Retriever.load(tmp_path, embed_fn=fake_embed)
    assert [c.text for c in loaded.chunks] == [c.text for c in CHUNKS]
    assert loaded.search("printer", k=1)[0][0].source == "printer.md"