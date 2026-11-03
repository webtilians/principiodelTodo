"""Embeddings saved by scripts/fetch_embeddings.py, served offline.

Real vectors for every text the development runs embed, so calibration needs no
API key. A text missing from the file falls back to a hash vector and is
counted, so a run can report how much of it was real.
"""
import json
from pathlib import Path

from src.infinito3.persistent_memory import HashEmbeddingProvider


class ReplayEmbeddings:
    def __init__(self, base: str):
        import numpy as np

        base = Path(base)
        texts = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))["texts"]
        self.index = {text: i for i, text in enumerate(texts)}
        self.matrix = np.load(base.with_suffix(".npy"))
        self.fallback = HashEmbeddingProvider(dimensions=self.matrix.shape[1])
        self.misses = 0

    def prefetch(self, texts) -> None:
        pass

    def embed(self, text: str):
        row = self.index.get(text)
        if row is None:
            self.misses += 1
            return self.fallback.embed(text)
        return self.matrix[row].astype(float).tolist()
