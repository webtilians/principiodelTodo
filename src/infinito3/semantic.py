"""Ranking memories by meaning, so "what music do I like?" finds "my favourite songs are...".

The hashed embeddings the store keeps match words. A question often shares none with the memory that
answers it: on one person's real memory they put the answer among the 20 closest for 70% of everyday
questions; this static multilingual model (model2vec's potion-multilingual-128M, quantized to int8,
122 MB of RAM, no GPU) does it for 95%, in milliseconds, on a one-CPU server.

The model is read from a local folder (INFINITO_EMBEDDINGS), never downloaded at run time, and the
memories are ranked in this process only: nothing leaves the server. Needs the `semantic` extra.

    python -m infinito3.semantic save <folder>   # once, where there is RAM: downloads and quantizes it
"""
from __future__ import annotations

import sys
import threading
from typing import List, Optional, Sequence

MODEL = "minishlab/potion-multilingual-128M"


class MeaningRanker:
    """Orders texts by how close their meaning is to a query. Loads its model on first use."""

    def __init__(self, folder: str):
        self.folder = folder
        self._model = None
        self._lock = threading.Lock()

    def _loaded(self):
        with self._lock:
            if self._model is None:
                from model2vec import StaticModel

                self._model = StaticModel.from_pretrained(self.folder)
            return self._model

    def warm(self) -> None:
        """Load the model now (in a thread at startup) instead of on the first question."""
        self._loaded()

    def rank(self, query: str, texts: Sequence[str]) -> List[int]:
        """Indexes of texts, closest in meaning to the query first."""
        import numpy as np

        if not texts:
            return []
        vectors = self._loaded().encode([query, *texts]).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
        return [int(i) for i in np.argsort(-(vectors[1:] @ vectors[0]), kind="stable")]


def ranker_from(folder: Optional[str]) -> Optional[MeaningRanker]:
    """A ranker when the folder holds a model and model2vec is installed, else None (word search stays)."""
    from pathlib import Path

    if not folder or not (Path(folder) / "model.safetensors").exists():
        return None
    try:
        import model2vec  # noqa: F401
    except ImportError:
        return None
    return MeaningRanker(folder)


def save(folder: str) -> None:
    from model2vec import StaticModel

    StaticModel.from_pretrained(MODEL, quantize_to="int8").save_pretrained(folder)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        save(sys.argv[2])
        print(f"Saved {MODEL} (int8) in {sys.argv[2]}.")
    else:
        raise SystemExit(__doc__)
