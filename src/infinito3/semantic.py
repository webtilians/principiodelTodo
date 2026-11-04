"""Ranking memories by meaning, so "what music do I like?" finds "my favourite songs are...".

The hashed embeddings the store keeps match words. A question often shares none with the memory that
answers it: on one person's real memory they put the answer among the 20 closest for 70% of everyday
questions; the static multilingual model potion-multilingual-128M (model2vec, int8) does it for 95%.

It runs here without model2vec or the tokenizers library: on a small server their tokenizer alone took
about 490 MB. The model is a table of 500,000 word pieces with a vector each; a sentence's vector is
the mean of the vectors of its pieces. This module splits text into those pieces itself (the same
Unigram segmentation, from a compact list of pieces) and reads the vectors straight from the model
file through a memory map, so the system can drop those pages whenever it needs the RAM. Loaded, it
holds about 100 MB, and a recall over 150 memories takes milliseconds.

The model is read from a local folder (INFINITO_EMBEDDINGS), never downloaded at run time, and nothing
leaves the process.

    python -m infinito3.semantic save <folder>   # once, on a computer with RAM to spare (needs model2vec)
"""
from __future__ import annotations

import json
import struct
import sys
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

MODEL = "minishlab/potion-multilingual-128M"
PIECES = "pieces.txt"  # one word piece per line, in vocabulary order
SCORES = "scores.f32"  # each piece's Unigram log-probability, float32, same order
SPACE = "▁"  # how the tokenizer writes the space before a word


def _matrix(path: Path):
    """The embeddings in a safetensors file, memory-mapped (int8 rows)."""
    import numpy as np

    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    entry = header["embeddings"]
    dtype = {"I8": np.int8, "F16": np.float16, "F32": np.float32}[entry["dtype"]]
    start, _ = entry["data_offsets"]
    return np.memmap(path, dtype=dtype, mode="r", offset=8 + size + start, shape=tuple(entry["shape"]))


class MeaningRanker:
    """Orders texts by how close their meaning is to a query. Loads its model on first use."""

    def __init__(self, folder: str, cache_size: int = 4096):
        self.folder = Path(folder)
        self._lock = threading.Lock()
        self._ids: Optional[Dict[str, int]] = None
        self._cache: "OrderedDict[str, object]" = OrderedDict()  # a memory's vector, computed once
        self._cache_size = cache_size

    def _load(self) -> None:
        import numpy as np

        with self._lock:
            if self._ids is not None:
                return
            pieces = (self.folder / PIECES).read_text(encoding="utf-8").split("\n")
            self._scores = np.fromfile(self.folder / SCORES, dtype=np.float32)
            self._ids = {piece: i for i, piece in enumerate(pieces)}
            self._longest = max(len(p) for p in pieces)
            self._unknown = self._ids.get("[UNK]", 1)
            self._unknown_score = float(self._scores.min()) - 10.0
            self._vectors = _matrix(self.folder / "model.safetensors")

    def warm(self) -> None:
        """Load now (in a thread at startup) instead of on the first question."""
        self._load()

    def pieces(self, text: str) -> List[int]:
        """The ids of the word pieces of text, by Unigram's most likely segmentation."""
        self._load()
        text = SPACE + SPACE.join(unicodedata.normalize("NFKC", text).split())
        n = len(text)
        best = [0.0] + [float("-inf")] * n
        back = [0] * (n + 1)
        chosen = [self._unknown] * (n + 1)
        ids, scores, longest = self._ids, self._scores, self._longest
        for end in range(1, n + 1):
            for start in range(max(0, end - longest), end):
                if best[start] == float("-inf"):
                    continue
                piece = ids.get(text[start:end])
                if piece is not None:
                    score = best[start] + float(scores[piece])
                    if score > best[end]:
                        best[end], back[end], chosen[end] = score, start, piece
            if best[end] == float("-inf"):  # a character no piece covers
                best[end], back[end], chosen[end] = best[end - 1] + self._unknown_score, end - 1, self._unknown
        found, end = [], n
        while end > 0:
            found.append(chosen[end])
            end = back[end]
        return [i for i in reversed(found) if i != self._unknown]

    def vector(self, text: str):
        import numpy as np

        cached = self._cache.get(text)
        if cached is not None:
            self._cache.move_to_end(text)
            return cached
        ids = self.pieces(text)
        mean = (self._vectors[ids].astype(np.float32).mean(axis=0) if ids
                else np.zeros(self._vectors.shape[1], dtype=np.float32))
        mean /= np.linalg.norm(mean) + 1e-9
        self._cache[text] = mean
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return mean

    def rank(self, query: str, texts: Sequence[str]) -> List[int]:
        """Indexes of texts, closest in meaning to the query first."""
        import numpy as np

        if not texts:
            return []
        matrix = np.stack([self.vector(t) for t in texts])
        return [int(i) for i in np.argsort(-(matrix @ self.vector(query)), kind="stable")]


def ranker_from(folder: Optional[str]) -> Optional[MeaningRanker]:
    """A ranker when the folder holds a saved model and numpy is installed, else None (word search stays)."""
    if not folder or not all((Path(folder) / name).exists() for name in ("model.safetensors", PIECES, SCORES)):
        return None
    try:
        import numpy  # noqa: F401
    except ImportError:
        return None
    return MeaningRanker(folder)


def save(folder: str) -> None:
    """Download the model, quantize it to int8 and write it with its compact list of pieces."""
    import numpy as np
    from model2vec import StaticModel

    StaticModel.from_pretrained(MODEL, quantize_to="int8").save_pretrained(folder)
    vocab = json.loads((Path(folder) / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    (Path(folder) / PIECES).write_text("\n".join(piece for piece, _ in vocab), encoding="utf-8")
    np.array([score for _, score in vocab], dtype=np.float32).tofile(Path(folder) / SCORES)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        save(sys.argv[2])
        print(f"Saved {MODEL} (int8) in {sys.argv[2]}.")
    else:
        raise SystemExit(__doc__)
