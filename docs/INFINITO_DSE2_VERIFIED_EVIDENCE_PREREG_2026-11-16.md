# DSE-2 · Prospective follow-up to DSE-1, 2026-11-16

DSE-1 (run 43947464211) was 0/6 for A,B,C because Qwen2.5-0.5B returned invalid output schemas 18/18, and none of the raw lists comprised a successful hidden-inversion plan. That negative result is FINAL for DSE-1 and its heldout cases must NOT be rerun or relabeled as confirmations.

## New question and design
Can an instruction-tuned Qwen2.5-1.5B model perform the **same unseen composition family** with better utility from observation-grounded **verified effect summaries** than from raw episodes or no memory? Exploratory new study, not confirmation; changes relative to DSE-1 are explicitly: different model size, new sealed heldout cases, schema normalization permitting either a JSON object `{"plan":[...]}` or a bare JSON array `[...]`, and explicit attention mask.

Training experiences and simulator semantics match DSE-1, newly recreated from tool events in a fresh in-memory **real INFINITO SQLite** memory store (T0..T6, 2 public-copy, 2 public-flip, 2 hidden-copy, 1 precondition failure). The unseen complete plan select-hidden/read-hidden/invert/verify MUST NOT occur during TRAIN. There are no private truth labels in memory and no direct knowledge of heldout hidden bits in prompt.

Three arms each see equal request/goal/tool catalogue, no prior conversations, same frozen Qwen2.5-1.5B checkpoint, greedy decoding, budget 5 actions and at most 96 new tokens. A: no references. B: retrieved raw episodes. C: same public episodes with mechanical provenance-checked effect descriptions; C's extraction is human-engineered code, not model learning. Prompt token budgets are **measured but not exactly equal** (interpretation is accordingly limited). Explicitly report outputs, exact model revision SHA when possible, per-arm verifier results, and tokens.

**New, sealed heldout:** D2H0..D2H5, (public, hidden) = (1,0), (0,1), (1,1), (0,0), (1,0), (0,1). Services = `grid-quiet`, `flux-pilot`, `amber-queue`, `pulse-map`, `canyon-signal`, `reef-cache`. These repeat one composition family; not six independent skills.

## Gates and stop rules
- 18 real model completions. Parse both accepted shapes as a 1..5 action plan ending with verify, tool names drawn strictly from catalogue. Never repair or supplement actions. Independent simulator verifies success (an unverified answer is a failure).
- Compare successes A/B/C and invalid outputs. Exploratory positive signal requires C >= A+2 and C >= B+2 with no integrity error.
- Never claim improvement in user's local Qwen 27B, learning of model weights, or universal memory benefit. If this fails, STOP rather than retune on D2H0..D2H5.
- Isolated ephemeral CPU CI, no paid model APIs, user data, production changes, real tools or secrets.
- Code and CI follow this protocol commit; no results are known at pre-registration time.
