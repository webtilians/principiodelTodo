#!/usr/bin/env python3
"""Fetch real embeddings for the development data, once, to calibrate offline.

Embeds every text that INFINITO would embed on two development sets, and
nothing from the sets used to report results:

- the memory-updates benchmark on its ``dev`` partition;
- Bemula's development instances: the full benchmark minus Bemula-Lite, which
  is kept for evaluation. User turns of both dialogue variants, the forget
  requests, the probe questions and the relearning turns.

Writes ``<out>.json`` (the texts) and ``<out>.npy`` (float16 vectors, one row
per text). Launched by .github/workflows/infinito-embeddings-fetch.yml.

Usage: python scripts/fetch_embeddings.py --bemula path/to/bemula-benchmark --out results/dev-embeddings
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark_memory_updates import EMBEDDING_MODEL, PARTITIONS, make_scenario, texts_to_embed  # noqa: E402


def updates_texts(partition: str) -> set:
    bases = PARTITIONS[partition]
    texts = set()
    for set_name, base in bases.items():
        for i in range(4):
            for j in range(6):
                texts |= texts_to_embed(make_scenario(set_name, base + i * 1000 + j))
    return texts


def bemula_dev_texts(root: Path) -> set:
    load = lambda p: [json.loads(line) for line in p.open(encoding="utf-8") if line.strip()]
    lite = {i["instance_id"] for i in load(root / "data/lite/dialogues.jsonl")}
    dev = [i for i in load(root / "data/full/dialogues.jsonl") if i["instance_id"] not in lite]
    dev_ids = {i["instance_id"] for i in dev}
    texts = set()
    for instance in dev:
        for variant in ("sessions", "control_sessions"):
            for session in instance[variant]:
                texts.update(t["text"] for t in session["turns"] if t["speaker"] == "user")
        texts.update(r["request_text"] for r in instance["forget_requests"])
    for item in load(root / "data/full/questions.jsonl"):
        if item["instance_id"] in dev_ids:
            texts.add(item["question"])
            texts.update(t["text"] for t in item.get("context_turns", []) if t["speaker"] == "user")
    assert not dev_ids & lite
    return texts


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bemula", required=True, help="clone of github.com/Nboxff/bemula-benchmark")
    parser.add_argument("--partition", default="dev")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    texts = sorted(updates_texts(args.partition) | bemula_dev_texts(Path(args.bemula)))
    print(f"{len(texts)} texts, {sum(len(t) for t in texts)} characters", flush=True)

    from openai import OpenAI
    client = OpenAI(timeout=60, max_retries=6)
    rows, batch, size = [], [], 0
    def flush():
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
    for text in texts:
        # The endpoint takes at most 2048 inputs and about 300k tokens per request.
        if batch and (len(batch) >= 512 or size + len(text) > 400_000):
            flush()
            batch, size = [], 0
        batch.append(text)
        size += len(text)
    if batch:
        flush()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), np.asarray(rows, dtype=np.float16))
    out.with_suffix(".json").write_text(json.dumps({"model": EMBEDDING_MODEL, "texts": texts},
                                                   ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(rows)} vectors to {out.with_suffix('.npy')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
