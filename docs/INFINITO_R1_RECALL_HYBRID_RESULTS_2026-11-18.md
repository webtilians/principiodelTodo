# R1: hybrid retrieval for MCP `recall` (results)

Protocol: [INFINITO_R1_RECALL_HYBRID_PROTOCOL_2026-11-18.md](INFINITO_R1_RECALL_HYBRID_PROTOCOL_2026-11-18.md),
committed in `5f59b0b` before the code and the data. Code: `observatory/r1_recall_hybrid.py`. Raw results:
`artifacts/r1/dev.json`, `artifacts/r1/confirm.json`. Run on 2026-11-18 on the maintainer's PC with the model
production uses (potion-multilingual-128M int8, copied from the server), SQLite 3.45.1. No paid model, no
network, no personal data.

## Verdict

**No hybrid arm passes the gate. Production stays as it is and the FTS5 line is closed**, as the protocol
says for this result.

Both hybrids find exact identifiers that the meaning ranker misses once the memory is large, but on the
CONFIRM split they lose paraphrase questions and gain nothing at 200 memories.

## CONFIRM (run once), 126 questions

Recall@20 and MRR@10 over all questions; p95 is milliseconds per question, warm.

| Arm | N=200 R@20 | N=200 MRR | N=2000 R@20 | N=2000 R@5 | N=2000 MRR | N=2000 p95 |
|---|---:|---:|---:|---:|---:|---:|
| **A** meaning ranker (production, RAG baseline) | 0.960 | 0.678 | 0.841 | 0.603 | 0.363 | 2.2 |
| B store.search (fallback) | 0.857 | 0.465 | 0.841 | 0.460 | 0.370 | 48.6 |
| C FTS5 + BM25 | 0.865 | 0.620 | 0.841 | 0.722 | 0.543 | 0.8 |
| D RRF(A, C) | 0.968 | 0.664 | **0.905** | **0.762** | **0.498** | 3.4 |
| E RRF(A, B) | 0.960 | 0.588 | 0.897 | 0.698 | 0.388 | 72.9 |

Recall@20 by family at N = 2,000:

| Arm | F1 identifiers (40) | F2 client and topic (36) | F3 paraphrase (20) | F4 changed fact (30) |
|---|---:|---:|---:|---:|
| A | 0.75 | 0.94 | 0.60 | 1.00 |
| B | 1.00 | 1.00 | 0.00 | 1.00 |
| C | 1.00 | 0.97 | 0.05 | 1.00 |
| D | 1.00 | 0.94 | 0.50 | 1.00 |
| E | 1.00 | 0.94 | 0.45 | 1.00 |

## Gate

| Criterion | D | E |
|---|---|---|
| 1. R@20 ≥ A + 3 points at N=200 and N=2000 | **fails** at 200 (+0.8); passes at 2000 (+6.4) | **fails** at 200 (+0.0); passes at 2000 (+5.6) |
| 2. No family more than 2 points below A at N=2000 | **fails**: F3 −10 (2 of 20 questions) | **fails**: F3 −15 |
| 3. MRR@10 ≥ A − 0.01 at both sizes | **fails** at 200 (−0.014) | **fails** at 200 (−0.090) |
| 4. p95 at N=2000 ≤ max(2 × A, A + 20 ms) = 22.2 ms | passes (3.4) | **fails** (72.9) |
| 5. FTS5 deletion check | passes | — |

## What was learned

- **The meaning ranker misses exact identifiers in a large memory.** At N = 2,000 it put the right ticket or
  PR among the 20 shown for 75% of questions that named one; every arm with a lexical ranking found 100%.
  At N = 200 it found all of them.
- **Lexical ranking alone cannot find paraphrases.** B and C found 0–5% of F3; fusing them with the meaning
  ranker kept most of A's paraphrase hits but lost 2 or 3 of 20.
- **For a changed fact, the later value is found but not ranked first.** Every arm put the later value among
  the 20 shown in every case; A and D ranked it above the earlier one in 20% of them, B, C and E in none.
  `recall` lists what it shows oldest first, so the later value is still the last one the assistant reads.
- **At today's sizes nothing is slow.** A is about 2 ms per question at 2,000 memories; `store.search` is the
  slowest part of the fallback (about 46 ms).
- **FTS5 is available** on this PC and on the server (SQLite 3.45.1 both), and with `secure-delete` a deleted
  row's unique word did not remain in the file. Not needed now.

## Harness changes and weaknesses, as found

- After the first DEV run at N = 200, latency was unfair: the meaning ranker caches the vector of each
  question, so arm A paid for it and D and E reused it. Fixed before the full DEV and CONFIRM runs by clearing
  the question vectors before each arm. No arm, parameter or datum changed.
- F2 draws (client, topic) pairs from one pool in both splits, so 19 of 36 F2 questions repeat between DEV
  and CONFIRM with other answers. The protocol did not promise otherwise for F2 and nothing was tuned on DEV,
  but CONFIRM is less independent for that family. The F2 template also writes "sobre el facturación" and
  "sobre el reuniones" for two topics; the same wording reached every arm.
- F3 has 20 questions per split, so one question is 5 points. The F3 losses are 2 and 3 questions.
- The data is synthetic and written by one author; latency is this PC's, without the server's decryption.

- N = 10,000 was not measured: building it through `AgentMemory.remember` passed the protocol's 15-minute
  limit (2,000 took 70 s; the run was stopped at 15 minutes). The file `artifacts/r1/dev_200_probe.json` is
  the first DEV run, with the unfair latency described above; its accuracy equals the later run's.

## What next

Per the protocol, the next retrieval work targets something other than FTS5. The clearest weakness this run
found in production is the identifier miss at large sizes; whether a lexical signal used only when a question
names a code or number fixes it without hurting paraphrases is a **new hypothesis**, and would need its own
protocol and new data, not a re-run of this one.
