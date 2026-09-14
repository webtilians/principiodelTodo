#!/usr/bin/env python3
import math
import os

from openai import OpenAI


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")
    model = os.environ.get("INFINITO_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI(api_key=api_key)

    banks = (
        (
            "outdoor",
            "¿Qué actividades al aire libre te he dicho que me gustan? Incluye todas las que recuerdes.",
            "actividades al aire libre",
            (
                ("senderismo", "Me gusta hacer senderismo.", "hacer senderismo", True),
                ("escalada", "Me gusta la escalada.", "escalada", True),
                ("kayak", "Me gusta navegar en kayak.", "navegar en kayak", True),
                ("ajedrez", "Me gusta jugar al ajedrez.", "jugar al ajedrez", False),
                ("jazz", "Me gusta escuchar jazz.", "escuchar jazz", False),
                ("curry", "Me gusta cocinar curry.", "cocinar curry", False),
                ("telescopio", "Me encanta observar estrellas con telescopio.", "observar estrellas con telescopio", True),
                ("ceramica", "Me gusta la cerámica japonesa.", "cerámica japonesa", False),
            ),
        ),
        (
            "creative",
            "¿Qué aficiones creativas o artísticas recuerdas que me gustan?",
            "aficiones creativas artísticas",
            (
                ("ceramica", "Me gusta hacer cerámica.", "hacer cerámica", True),
                ("fotografia", "Me gusta la fotografía nocturna.", "fotografía nocturna", True),
                ("pintura", "Me gusta pintar con acuarelas.", "pintar con acuarelas", True),
                ("running", "Me gusta salir a correr.", "salir a correr", False),
                ("ajedrez", "Me gusta jugar al ajedrez.", "jugar al ajedrez", False),
                ("curry", "Me gusta cocinar curry.", "cocinar curry", False),
                ("kayak", "Me gusta navegar en kayak.", "navegar en kayak", False),
            ),
        ),
        (
            "water",
            "¿Qué actividades relacionadas con el agua recuerdas que me gustan?",
            "actividades relacionadas con el agua",
            (
                ("kayak", "Me gusta navegar en kayak.", "navegar en kayak", True),
                ("natacion", "Me gusta nadar en el mar.", "nadar en el mar", True),
                ("buceo", "Me gusta practicar buceo.", "practicar buceo", True),
                ("escalada", "Me gusta la escalada.", "escalada", False),
                ("jazz", "Me gusta escuchar jazz.", "escuchar jazz", False),
                ("ceramica", "Me gusta hacer cerámica.", "hacer cerámica", False),
            ),
        ),
    )

    for bank_name, query, semantic_core, candidates in banks:
        texts = [query, semantic_core]
        texts += [text for _, text, _, _ in candidates]
        texts += [value for _, _, value, _ in candidates]
        response = client.embeddings.create(model=model, input=texts)
        vectors = [item.embedding for item in response.data]
        qvec, core_vec = vectors[:2]
        full_vectors = vectors[2 : 2 + len(candidates)]
        value_vectors = vectors[2 + len(candidates) :]

        full_scored = []
        core_scored = []
        for candidate, full_vec, value_vec in zip(candidates, full_vectors, value_vectors):
            name, text, value, relevant = candidate
            full_scored.append((cosine(qvec, full_vec), name, relevant, text))
            core_scored.append((cosine(core_vec, value_vec), name, relevant, value))

        full_scored.sort(reverse=True)
        core_scored.sort(reverse=True)
        print(f"BANK={bank_name} MODE=full_query_full_memory")
        for rank, (score, name, relevant, text) in enumerate(full_scored, 1):
            print(f"{rank}\t{score:.6f}\t{'R' if relevant else 'D'}\t{name}\t{text}")
        print(f"BANK={bank_name} MODE=semantic_core_fact_value")
        for rank, (score, name, relevant, value) in enumerate(core_scored, 1):
            print(f"{rank}\t{score:.6f}\t{'R' if relevant else 'D'}\t{name}\t{value}")
        print()


if __name__ == "__main__":
    main()
