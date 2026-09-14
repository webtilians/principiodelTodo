#!/usr/bin/env python3
import os

from openai import OpenAI

from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker
from src.infinito3.types import ContextItem, ContextSource


BANKS = (
    (
        "creative",
        "¿Qué aficiones creativas o artísticas te he dicho que me gustan? Incluye todas las que recuerdes.",
        (
            ("acuarelas", "pintar con acuarelas", True),
            ("ceramica", "hacer cerámica", True),
            ("fotografia", "fotografía nocturna", True),
            ("ajedrez", "jugar al ajedrez", False),
            ("curry", "cocinar curry", False),
            ("kayak", "navegar en kayak", False),
            ("running", "salir a correr", False),
        ),
    ),
    (
        "water",
        "¿Qué actividades relacionadas con el agua recuerdas que me gustan? Incluye todas.",
        (
            ("nadar", "nadar en el mar", True),
            ("buceo", "practicar buceo", True),
            ("kayak", "navegar en kayak", True),
            ("acuarelas", "pintar con acuarelas", False),
            ("ceramica", "hacer cerámica", False),
            ("ajedrez", "jugar al ajedrez", False),
        ),
    ),
    (
        "sports",
        "¿Qué deportes recuerdas que me gustan?",
        (
            ("nadar", "nadar", True),
            ("running", "correr por montaña", True),
            ("scifi", "leer ciencia ficción", False),
            ("cafe", "tomar café", False),
        ),
    ),
)


def main() -> int:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")
    model = os.environ.get("INFINITO_RERANKER_MODEL", "gpt-5.6-luna")
    client = OpenAI(api_key=api_key)
    reranker = LLMSemanticMembershipReranker(
        OpenAIResponsesAdapter(client, model=model, reasoning_effort=None)
    )

    failures = 0
    total_tokens = 0
    for bank_name, query, rows in BANKS:
        candidates = [
            ContextItem(
                source=ContextSource.USER_MODEL,
                content=f"Me gusta {fact}.",
                score=1.0,
                estimated_tokens=8,
                memory_id=name,
                metadata={"fact_value": fact, "fact_predicate": "likes"},
            )
            for name, fact, _ in rows
        ]
        expected = {name for name, _, relevant in rows if relevant}
        result = reranker.rerank(query, candidates)
        selected = set(result.selected_ids)
        ok = result.success and selected == expected
        total_tokens += int(result.usage.get("total_tokens") or 0)
        failures += int(not ok)
        print(
            f"bank={bank_name} success={result.success} ok={ok} "
            f"selected={sorted(selected)} expected={sorted(expected)} "
            f"tokens={result.usage.get('total_tokens', 0)} error={result.error or 'none'}"
        )

    print(f"banks={len(BANKS)} failures={failures} total_tokens={total_tokens}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
