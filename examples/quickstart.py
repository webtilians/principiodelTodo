"""INFINITO quickstart: runs offline, no API key, standard library only.

It shows what the memory layer does today:
1. secrets are blocked before they reach memory or the LLM;
2. durable facts persist across sessions in SQLite;
3. a newer exclusive fact supersedes the old one and keeps its history;
4. the LLM receives retrieved memory as untrusted reference data.

Run from the repository root:  python examples/quickstart.py
"""
import sys
import tempfile
from pathlib import Path

try:
    from infinito3 import CognitiveEngine, CognitiveLoop, RecordingLLMAdapter
except ImportError:  # plain checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from infinito3 import CognitiveEngine, CognitiveLoop, RecordingLLMAdapter


def describe(decision) -> str:
    if decision.gate is None:
        return f"BLOCKED ({decision.safety.reason}), never stored"
    if decision.stored_memory_id:
        return "stored"
    return "not stored"


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = str(Path(tmp) / "memory.db")

        print("Session 1")
        engine = CognitiveEngine.persistent(db_path=db_path)
        for text in (
            "Me llamo Ana.",
            "Vivo en Madrid.",
            "Mi color favorito es el azul.",
            "Mi contraseña es hunter2",
        ):
            print(f"  {text:32} -> {describe(engine.process(text))}")
        engine.memory_store.close()

        print("\nSession 2 (new engine, same database)")
        engine = CognitiveEngine.persistent(db_path=db_path)
        text = "Ahora vivo en Valencia."
        print(f"  {text:32} -> {describe(engine.process(text))}")
        print("  location history:")
        for record in engine.memory_store.history_for_fact("user", "location"):
            print(f"    {record.fact_value:10} {record.status.value}")

        loop = CognitiveLoop(engine, RecordingLLMAdapter())
        run = loop.turn("¿En qué ciudad vivo ahora?")
        print("\nWhat the LLM receives:")
        for message in run.request.messages[1:]:
            print(f"  [{message.role}] " + message.content.replace("\n", "\n    "))
        engine.memory_store.close()


if __name__ == "__main__":
    main()
