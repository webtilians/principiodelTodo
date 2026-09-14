#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.semantic_interpreter import SemanticCognitiveEventExtractor, SemanticStateQueryAnalyzer


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)
    adapter = OpenAIResponsesAdapter(client, model=model, reasoning_effort=None)
    clock = Clock(datetime(2026, 11, 9, 8, 0, 0))
    extractor = SemanticCognitiveEventExtractor(adapter, now_fn=clock)
    analyzer = SemanticStateQueryAnalyzer(adapter, max_output_tokens=220)

    # expected_types may contain schema-equivalent operations. Predicate=None
    # means downstream state reconciliation, not the parser, owns ontology alignment.
    event_cases = [
        ("He cambiado otra vez de bici: ahora uso una Yeti SB160.", {"replace_fact"}, "bike", "yeti sb160"),
        ("I no longer drink kombucha.", {"retract_preference", "retract_fact"}, None, "kombucha"),
        ("I have started enjoying bouldering.", {"assert_preference"}, "likes", "bouldering"),
        ("Mi gata se llama Nube.", {"assert_fact"}, "pet_name", "nube"),
        ("Mi frase de verificación es: las nubes cantan en hexadecimal.", {"store_note"}, "verification_phrase", "nubes cantan"),
        ("El domingo a las 17 tengo clase de guitarra.", {"create_goal"}, "goal", "guitarra"),
        ("La cita del veterinario cambia: ya no es el martes, será el jueves a las 20.", {"reschedule_goal"}, "goal", "veterin"),
        ("I already finished the server backup; mark it completed.", {"complete_goal"}, "goal", "backup"),
        ("Cancel Friday's property-tax payment; it is no longer needed.", {"cancel_goal"}, "goal", "tax"),
        ("I picked up the glasses already; close that task.", {"complete_goal"}, "goal", "glass"),
        ("La clase de guitarra se cancela; no voy a ir.", {"cancel_goal"}, "goal", "guitarra"),
    ]

    failures = []
    rows = []
    for text, expected_types, expected_predicate, value_fragment in event_cases:
        events = extractor.extract(text)
        compact = [
            {
                "type": event.type.value,
                "predicate": event.predicate,
                "value": event.value,
                "due_at": event.due_at.isoformat() if event.due_at else None,
                "previous_due_at": event.metadata.get("previous_due_at"),
                "confidence": event.confidence,
            }
            for event in events
        ]
        matching = [
            event for event in events
            if event.type.value in expected_types
            and (expected_predicate is None or event.predicate == expected_predicate)
            and value_fragment in (event.value or "").lower()
        ]
        ok = bool(matching)
        if not ok:
            failures.append({"text": text, "expected_types": sorted(expected_types), "events": compact})
        rows.append({"text": text, "ok": ok, "events": compact})

    query_cases = [
        ("What are my current name, city, bike and language?", {"name", "location", "bike", "studying_language"}, False, None),
        ("Where did I live immediately before Utrecht?", set(), False, ("location", "utrecht")),
        ("Enumera todos mis compromisos de esta semana.", set(), True, None),
    ]
    query_rows = []
    for query, expected_predicates, expected_goals, expected_history in query_cases:
        plan = analyzer.analyze(query)
        history_pairs = {(item.predicate, (item.before_value or "").lower()) for item in plan.history}
        ok = expected_predicates.issubset(set(plan.predicates)) and plan.asks_goals == expected_goals
        if expected_history is not None:
            ok = ok and expected_history in history_pairs
        if not ok:
            failures.append({
                "query": query,
                "predicates": plan.predicates,
                "asks_goals": plan.asks_goals,
                "history": list(history_pairs),
            })
        query_rows.append({
            "query": query,
            "ok": ok,
            "predicates": plan.predicates,
            "asks_goals": plan.asks_goals,
            "history": list(history_pairs),
            "confidence": plan.confidence,
        })

    report = {
        "model": model,
        "event_cases": rows,
        "query_cases": query_rows,
        "event_usage": extractor.usage_summary(),
        "query_usage": analyzer.usage_summary(),
        "failures": failures,
    }
    out = Path("semantic-diagnostic-results")
    out.mkdir(exist_ok=True)
    (out / "semantic-interpreter-diagnostic.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
