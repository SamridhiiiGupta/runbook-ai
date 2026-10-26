import numpy as np
import ollama

from .chunker import Chunk

EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts: list[str]) -> np.ndarray:
    """Turn a list of texts into a matrix: one row of 768 numbers per text."""
    response = ollama.embed(model=EMBED_MODEL, input=texts)
    return np.array(response["embeddings"], dtype=float)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid dividing by zero for empty vectors
    return matrix / norms


class Retriever:
    """Finds the chunks whose meaning is closest to a question."""

    def __init__(self, chunks: list[Chunk], embed_fn=embed_texts):
        if not chunks:
            raise ValueError("Retriever needs at least one chunk")
        self.chunks = chunks
        self.embed_fn = embed_fn
        # This model works best with these prefixes (see its model card)
        docs = [f"search_document: {c.section}\n{c.text}" for c in chunks]
        self.matrix = _normalize(embed_fn(docs))

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, score) pairs, best first."""
        q = _normalize(self.embed_fn([f"search_query: {query}"]))[0]
        scores = self.matrix @ q                  # one score per chunk
        best = np.argsort(scores)[::-1][:k]       # indexes of the highest scores
        return [(self.chunks[i], float(scores[i])) for i in best]