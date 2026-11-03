#!/usr/bin/env python3
"""Run INFINITO on Bemula, a benchmark of forgetting on request (arXiv 2610.18052).

INFINITO is wrapped in Bemula's own MemoryAdapter and scored by Bemula's own
runner, LLM judge and metrics, unchanged, from a clone of
github.com/Nboxff/bemula-benchmark at commit 862aff6. The answer prompt is the
one Bemula's RAG baseline uses; only the context differs.

INFINITO stores what the user says (not the assistant's replies), recognizes
forget requests with rules, lets an LLM pick the memories that hold what is to
be forgotten, and deletes them.

Splits: lite (31 instances: the evaluation set), dev (the other 19 instances of
the full benchmark, for development), quick_check (1 instance).

Offline (--offline): no API key. Embeddings are hash vectors or a replay file
from scripts/fetch_embeddings.py; the reader answers with its context verbatim
and Bemula's MockJudge checks whether the protected value is in it. The planner
is the overlap planner or, on dev, an oracle that knows which turns hold each
target fact. This measures whether evidence arrives and whether forgotten
values are gone, not how a model answers.

Real: gpt-5.6-luna answers and plans (reasoning off), text-embedding-3-small
embeds, and Bemula's judge, DeepSeek-V4.1-Flash, grades, as in the paper's
GPT-5.6-Luna runs. Needs OPENAI_API_KEY and DEEPSEEK_API_KEY.

Usage:
  python scripts/bemula_infinito.py --bemula ../bemula --split dev --offline --limit 2
  python scripts/bemula_infinito.py --bemula ../bemula --split lite --workers 8 --out results/bemula-lite
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.infinito3.engine import CognitiveEngine  # noqa: E402
from src.infinito3.forgetting import ForgetPlan, LLMForgetPlanner, OverlapForgetPlanner  # noqa: E402
from src.infinito3.persistent_memory import HashEmbeddingProvider  # noqa: E402
from src.infinito3.types import LLMResponse  # noqa: E402
from scripts.replay_embeddings import ReplayEmbeddings  # noqa: E402

AGENT_MODEL = "gpt-5.6-luna"
JUDGE_MODEL = "deepseek-flash"
EMBEDDING_MODEL = "text-embedding-3-small"
BEMULA_COMMIT = "862aff6be384"
# Chosen on the development split (offline, before any real run): Bemula asks
# open questions about long conversations, so the context keeps the 16 most
# relevant memories, as a RAG would, instead of pruning to the dominant ones.
TOP_K = 16
CONTEXT_BUDGET = 3000
PRUNE = False
FORGET_CANDIDATES = None  # every memory
NO_MEMORY = "I don't have that information."
# Bemula's RAG baseline prompt (bemula/baselines/rag.py), word for word.
ANSWER_PROMPT = (
    "You are a personal assistant with access to remembered conversation "
    "history. Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't have that "
    "information. Keep the answer short.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def import_bemula(path: Path):
    """Bemula's modules, from its clone; its OpenAI-compatible calls go to OpenAI or DeepSeek by model."""
    sys.path.insert(0, str(path))
    import bemula.llm as llm
    from bemula.baselines.base import MemoryAdapter
    from bemula.eval import judge, metrics, runner
    from bemula import schemas

    def resolve_endpoint(model):
        if "deepseek" in (model or "").lower():
            key = os.environ.get("DEEPSEEK_API_KEY")
            if not key:
                raise RuntimeError("DEEPSEEK_API_KEY is not set")
            return os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"), key
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return None, key

    llm.resolve_endpoint = resolve_endpoint
    return llm, MemoryAdapter, judge, metrics, runner, schemas


class BatchEmbeddings:
    """Real embeddings, fetched in batches and shared by every worker."""

    def __init__(self, client):
        self.client, self.cache, self.requests = client, {}, 0
        self._lock = threading.Lock()

    def prefetch(self, texts) -> None:
        with self._lock:
            missing = sorted({t for t in texts if t and t not in self.cache})
        batch, size = [], 0
        for text in missing + [None]:
            if batch and (text is None or len(batch) >= 512 or size + len(text) > 400_000):
                response = self.client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                with self._lock:
                    self.requests += 1
                    for item in response.data:
                        self.cache[batch[item.index]] = list(item.embedding)
                batch, size = [], 0
            if text is not None:
                batch.append(text)
                size += len(text)

    def embed(self, text: str):
        with self._lock:
            vector = self.cache.get(text)
        if vector is None:
            self.prefetch([text])
            with self._lock:
                vector = self.cache[text]
        return vector


class HashEmbeddings(HashEmbeddingProvider):
    def prefetch(self, texts) -> None:
        pass


class ContextEcho:
    """Offline reader: answers with the context it was given."""

    model = "context-echo"

    def chat(self, messages, **kwargs) -> str:
        prompt = messages[-1]["content"]
        return prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]


class BemulaChatAdapter:
    """INFINITO's LLMAdapter on top of a Bemula chat client, for the forget planner."""

    def __init__(self, llm):
        self.llm = llm

    def generate(self, request):
        messages = [{"role": "system" if m.role == "developer" else m.role, "content": m.content}
                    for m in request.messages]
        text = self.llm.chat(messages, temperature=0.0, max_tokens=request.max_output_tokens or 400,
                             task="infinito_forget_plan")
        return LLMResponse(text=text, provider="bemula-client", model=getattr(self.llm, "model", None))


class OraclePlanner:
    """Development only: forgets exactly the stored turns that mention a target fact."""

    def __init__(self):
        self.requests, self.turn_facts = {}, {}

    def set_instance(self, instance) -> None:
        self.requests = {r.request_text: r for r in instance.forget_requests}
        self.turn_facts = {t.text: set(t.linked_fact_ids)
                           for s in instance.sessions + instance.control_sessions for t in s.turns}

    def plan(self, request, candidates) -> ForgetPlan:
        target = self.requests.get(request)
        if target is None:
            return ForgetPlan(is_request=False)
        wanted = set(target.target_fact_ids)
        return ForgetPlan(forget_ids=[c.id for c in candidates if self.turn_facts.get(c.content, set()) & wanted],
                          remember=[f"My preference now is {target.replacement}."] if target.replacement else [])


def _parse_time(stamp):
    try:
        return datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return None


def make_adapter_class(MemoryAdapter):
    class InfinitoAdapter(MemoryAdapter):
        """INFINITO behind Bemula's adapter contract: reset, observe, query."""

        name = "infinito"

        def __init__(self, llm, embeddings, planner, questions, events, top_k, budget, builder_options=None):
            self.llm, self.embeddings, self.planner = llm, embeddings, planner
            self.questions, self.events = questions, events
            self.top_k, self.budget = top_k, budget
            self.builder_options = builder_options or {}
            self.engine, self.workdir, self.instance, self.variant = None, None, None, -1

        def begin_instance(self, instance) -> None:
            self.instance, self.variant = instance, -1
            texts = {t.text for s in instance.sessions + instance.control_sessions for t in s.turns
                     if t.speaker == "user"}
            for item in self.questions.get(instance.instance_id, []):
                texts.add(item.question)
                texts.update(t.text for t in item.context_turns if t.speaker == "user")
            self.embeddings.prefetch(texts)
            if hasattr(self.planner, "set_instance"):
                self.planner.set_instance(instance)

        def reset(self) -> None:
            self._close()
            self.variant += 1
            self.workdir = tempfile.mkdtemp(prefix="infinito-bemula-")
            self.engine = CognitiveEngine.persistent(
                db_path=str(Path(self.workdir) / "memory.db"), embedding_provider=self.embeddings,
                forget_planner=self.planner, forget_candidates=FORGET_CANDIDATES)
            for name, value in self.builder_options.items():
                setattr(self.engine.context_builder, name, value)

        def observe(self, turn) -> None:
            # INFINITO remembers what the user says, not the assistant's replies.
            if turn.speaker != "user":
                return
            decision = self.engine.process(turn.text, build_context=False, said_at=_parse_time(turn.timestamp))
            if decision.forget_request:
                requests = {r.request_text for r in self.instance.forget_requests}
                self.events.append({
                    "instance_id": self.instance.instance_id,
                    "variant": "control" if self.variant == 0 else "forget",
                    "is_benchmark_request": turn.text in requests,
                    "text": turn.text, "forgotten": decision.forgotten_count,
                    "stored_instead": len(decision.stored_memory_ids),
                })

        def query(self, question: str) -> str:
            packet = self.engine.recall(question, top_k=self.top_k, context_budget_tokens=self.budget)
            if not packet.items:
                return NO_MEMORY
            prompt = ANSWER_PROMPT.format(context=packet.rendered, question=question)
            return self.llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=512,
                                 task="infinito_answer").strip()

        def end_instance(self) -> None:
            self._close()

        def _close(self) -> None:
            if self.engine is not None:
                self.engine.memory_store.close()
                self.engine = None
            if self.workdir:
                shutil.rmtree(self.workdir, ignore_errors=True)
                self.workdir = None

    return InfinitoAdapter


def load_split(root: Path, split: str, schemas):
    load = lambda p, model: schemas.load_jsonl(root / p, model)
    if split == "dev":
        lite = {i.instance_id for i in load("data/lite/dialogues.jsonl", schemas.DialogueInstance)}
        instances = [i for i in load("data/full/dialogues.jsonl", schemas.DialogueInstance) if i.instance_id not in lite]
        ids = {i.instance_id for i in instances}
        items = [q for q in load("data/full/questions.jsonl", schemas.TestItem) if q.instance_id in ids]
        return instances, items
    folder = {"lite": "data/lite", "quick_check": "data/quick_check"}[split]
    return (load(f"{folder}/dialogues.jsonl", schemas.DialogueInstance),
            load(f"{folder}/questions.jsonl", schemas.TestItem))


def published_baselines(root: Path, ids, metrics, runner):
    """Bemula's published GPT-5.6-Luna runs, recomputed on the same instances."""
    rows = {}
    for run in sorted((root / "results").glob("*_full_gpt56_luna")):
        records = [runner.ItemRecord.model_validate_json(line)
                   for line in (run / "items.jsonl").open(encoding="utf-8") if line.strip()]
        subset = [r for r in records if r.instance_id in ids]
        if subset:
            rows[run.name.split("_full_")[0]] = metrics.compute_metrics(subset)
    return rows


def headline(m) -> dict:
    return {"Score": m["bemula_score"], "RPR": m["request_pass_rate"], "FE": m["FE"],
            "AR": m["AR"]["average"], "RA": m["RA"], "PR": m["PR"], "RLA": m["RLA"]}


def report(metrics_infinito, baselines, events, instances, meta) -> str:
    pct = lambda v: f"{100 * v:.1f}" if isinstance(v, (int, float)) else "-"
    lines = [f"## INFINITO on Bemula ({meta['split']}, {meta['mode']})", "",
             f"Commit `{meta['commit']}` · Bemula `{BEMULA_COMMIT}` · reader `{meta['reader']}` · "
             f"judge `{meta['judge']}` · planner `{meta['planner']}` · embeddings `{meta['embeddings']}` · "
             f"{meta['finished']}", "",
             "| System | Score | Requests passed | Forget efficacy | Attack robustness | Retention | Preservation | Relearning |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, m in [("**INFINITO**", metrics_infinito)] + sorted(baselines.items()):
        h = headline(m)
        lines.append(f"| {name} | {pct(h['Score'])} | {pct(h['RPR'])} | {pct(h['FE'])} | {pct(h['AR'])} | "
                     f"{pct(h['RA'])} | {pct(h['PR'])} | {pct(h['RLA'])} |")
    requests = sum(len(i.forget_requests) for i in instances)
    detected = len({e["text"] for e in events if e["is_benchmark_request"] and e["variant"] == "forget"})
    false_alarms = sum(1 for e in events if not e["is_benchmark_request"])
    lines += ["", "Baselines: Bemula's published GPT-5.6-Luna runs (same judge), recomputed on the same instances.",
              f"Forget requests recognized: {detected}/{requests}; other messages taken for requests and sent "
              f"to the planner: {false_alarms}. Memories deleted: "
              f"{sum(e['forgotten'] for e in events if e['variant'] == 'forget')}.", ""]
    return "\n".join(lines)


def commit_id() -> str:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bemula", required=True, help="clone of github.com/Nboxff/bemula-benchmark")
    parser.add_argument("--split", choices=("lite", "dev", "quick_check"), default="dev")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--embeddings", default=None, help="hash, openai, or replay:<path without suffix>")
    parser.add_argument("--planner", choices=("llm", "overlap", "oracle"), default=None)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--budget", type=int, default=CONTEXT_BUDGET)
    parser.add_argument("--prune", action=argparse.BooleanOptionalAction, default=PRUNE,
                        help="prune retrieved memories to the dominant ones (INFINITO's default elsewhere)")
    parser.add_argument("--judge-model", default=os.environ.get("BEMULA_JUDGE_MODEL", JUDGE_MODEL))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out", default="results/bemula")
    args = parser.parse_args(argv)

    root = Path(args.bemula)
    llm_module, MemoryAdapter, judge_module, metrics, runner, schemas = import_bemula(root)
    instances, items = load_split(root, args.split, schemas)
    if args.limit:
        instances = instances[: args.limit]
    ids = {i.instance_id for i in instances}
    questions = {}
    for item in items:
        if item.instance_id in ids:
            questions.setdefault(item.instance_id, []).append(item)
    if args.split == "lite" and (args.offline or args.planner == "oracle"):
        raise SystemExit("Bemula-Lite is the evaluation set: run it only for real, without the oracle.")

    embeddings_choice = args.embeddings or ("hash" if args.offline else "openai")
    planner_choice = args.planner or ("overlap" if args.offline else "llm")
    client = None
    if embeddings_choice == "openai":
        from openai import OpenAI
        client = OpenAI(timeout=60, max_retries=6)
        embeddings = BatchEmbeddings(client)
    elif embeddings_choice.startswith("replay:"):
        embeddings = ReplayEmbeddings(embeddings_choice.split(":", 1)[1])
    else:
        embeddings = HashEmbeddings(dimensions=256)

    if args.offline:
        reader, judge = ContextEcho(), judge_module.MockJudge()
    else:
        reader = llm_module.get_llm(model=AGENT_MODEL)
        judge = judge_module.LLMJudge(llm_module.get_llm(model=args.judge_model), temperature=0.0)
        # Fail fast on a missing key or model before any instance runs.
        print("agent:", reader.chat([{"role": "user", "content": "Reply only READY."}], temperature=0.0,
                                    max_tokens=16, task="preflight").strip()[:40], flush=True)
        print("judge:", judge.llm.chat([{"role": "user", "content": "Reply only READY."}], temperature=0.0,
                                       max_tokens=16, task="preflight").strip()[:40], flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items_path = out / "items.jsonl"
    items_path.unlink(missing_ok=True)
    write_lock = threading.Lock()

    def checkpoint(records) -> None:
        with write_lock, items_path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")

    def planner():
        if planner_choice == "llm":
            return LLMForgetPlanner(BemulaChatAdapter(reader))
        return OraclePlanner() if planner_choice == "oracle" else OverlapForgetPlanner()

    InfinitoAdapter = make_adapter_class(MemoryAdapter)
    events = []
    options = {"prune": args.prune}
    adapters = [InfinitoAdapter(reader, embeddings, planner(), questions, events, args.top_k, args.budget, options)
                for _ in range(max(1, args.workers))]
    runner.run_evaluation_parallel(adapters, instances, questions, judge, on_instance=checkpoint)

    records = [runner.ItemRecord.model_validate_json(line)
               for line in items_path.open(encoding="utf-8") if line.strip()]
    result = metrics.compute_metrics(records)
    baselines = published_baselines(root, ids, metrics, runner) if args.split != "quick_check" else {}
    meta = {"split": args.split, "mode": "offline" if args.offline else "real", "commit": commit_id(),
            "bemula_commit": BEMULA_COMMIT, "instances": len(instances), "items": len(records),
            "reader": getattr(reader, "model", "?"), "judge": "MockJudge" if args.offline else args.judge_model,
            "planner": planner_choice, "embeddings": EMBEDDING_MODEL if embeddings_choice == "openai" else embeddings_choice,
            "top_k": args.top_k, "context_budget": args.budget, "prune": args.prune,
            "forget_candidates": FORGET_CANDIDATES or "all",
            "finished": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
    if isinstance(embeddings, ReplayEmbeddings):
        meta["embedding_misses"] = embeddings.misses
    if isinstance(embeddings, BatchEmbeddings):
        meta["embedding_requests"] = embeddings.requests
    if not args.offline:
        meta["judge_failures"] = judge.failures

    (out / "metrics.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "baselines.json").write_text(json.dumps({k: headline(v) for k, v in baselines.items()}, indent=1),
                                        encoding="utf-8")
    (out / "forget_events.jsonl").write_text(
        "".join(json.dumps({**e, "text": e["text"] if e["is_benchmark_request"] else
                            hashlib.sha256(e["text"].encode()).hexdigest()[:12]}, ensure_ascii=False) + "\n"
                for e in events), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    summary = report(result, baselines, events, instances, meta)
    (out / "report.md").write_text(summary, encoding="utf-8")
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
