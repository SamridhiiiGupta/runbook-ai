import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import ollama

from .chunker import Chunk

EMBED_MODEL = "nomic-embed-text"


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Turn texts into a matrix: one row of numbers per text (in small batches)."""
    if not texts:
        return np.zeros((0, 0))
    parts = []
    for i in range(0, len(texts), batch_size):
        response = ollama.embed(model=EMBED_MODEL, input=texts[i:i + batch_size])
        parts.append(np.array(response["embeddings"], dtype=float))
    return np.vstack(parts)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1, so a dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1  # avoid dividing by zero for empty vectors
    return matrix / norms


class Retriever:
    """Finds the chunks whose meaning is closest to a question.

    Documents can be added, replaced and removed, and the index can be saved.
    """

    def __init__(self, chunks: list[Chunk] | None = None, embed_fn=embed_texts):
        self.embed_fn = embed_fn
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None
        if chunks:
            self.add_chunks(chunks)

    def _embed(self, chunks: list[Chunk]) -> np.ndarray:
        # This model works best with these prefixes (see its model card)
        docs = [f"search_document: {c.section}\n{c.text}" for c in chunks]
        return _normalize(self.embed_fn(docs))

    def _append(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if chunks:
            self._append(chunks, self._embed(chunks))

    def remove_source(self, source: str) -> int:
        """Remove every chunk of one document. Returns how many were removed."""
        keep = [i for i, c in enumerate(self.chunks) if c.source != source]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.matrix = self.matrix[keep] if keep else None
        return removed

    def replace_source(self, source: str, chunks: list[Chunk]) -> None:
        """Embed first, then swap, so a failure never loses the old version."""
        vectors = self._embed(chunks)
        self.remove_source(source)
        self._append(chunks, vectors)

    def sources(self) -> dict[str, int]:
        """Document name -> number of chunks."""
        counts: dict[str, int] = {}
        for c in self.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    def search(self, query: str, k: int = 3) -> list[tuple[Chunk, float]]:
        """Return the top-k (chunk, score) pairs, best first."""
        if self.matrix is None:
            return []
        q = _normalize(self.embed_fn([f"search_query: {query}"]))[0]
        scores = self.matrix @ q                  # one score per chunk
        best = np.argsort(scores)[::-1][:k]       # indexes of the highest scores
        return [(self.chunks[i], float(scores[i])) for i in best]

    def save(self, folder) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        data = [asdict(c) for c in self.chunks]
        (folder / "chunks.json").write_text(json.dumps(data), encoding="utf-8")
        vectors = folder / "vectors.npy"
        if self.matrix is not None:
            np.save(vectors, self.matrix)
        elif vectors.exists():
            vectors.unlink()

    @classmethod
    def load(cls, folder, embed_fn=embed_texts) -> "Retriever":
        folder = Path(folder)
        retriever = cls(embed_fn=embed_fn)
        meta, vectors = folder / "chunks.json", folder / "vectors.npy"
        if not meta.exists():
            return retriever
        chunks = [Chunk(**d) for d in json.loads(meta.read_text(encoding="utf-8"))]
        if chunks and vectors.exists():
            matrix = np.load(vectors)
            if len(matrix) == len(chunks):        # ignore a damaged index
                retriever.chunks, retriever.matrix = chunks, matrix
        return retriever