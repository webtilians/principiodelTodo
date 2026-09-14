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
            (
                ("senderismo", "Me gusta hacer senderismo.", True),
                ("escalada", "Me gusta la escalada.", True),
                ("kayak", "Me gusta navegar en kayak.", True),
                ("ajedrez", "Me gusta jugar al ajedrez.", False),
                ("jazz", "Me gusta escuchar jazz.", False),
                ("curry", "Me gusta cocinar curry.", False),
                ("telescopio", "Me encanta observar estrellas con telescopio.", True),
                ("ceramica", "Me gusta la cerámica japonesa.", False),
            ),
        ),
        (
            "creative",
            "¿Qué aficiones creativas o artísticas recuerdas que me gustan?",
            (
                ("ceramica", "Me gusta hacer cerámica.", True),
                ("fotografia", "Me gusta la fotografía nocturna.", True),
                ("pintura", "Me gusta pintar con acuarelas.", True),
                ("running", "Me gusta salir a correr.", False),
                ("ajedrez", "Me gusta jugar al ajedrez.", False),
                ("curry", "Me gusta cocinar curry.", False),
                ("kayak", "Me gusta navegar en kayak.", False),
            ),
        ),
        (
            "water",
            "¿Qué actividades relacionadas con el agua recuerdas que me gustan?",
            (
                ("kayak", "Me gusta navegar en kayak.", True),
                ("natacion", "Me gusta nadar en el mar.", True),
                ("buceo", "Me gusta practicar buceo.", True),
                ("escalada", "Me gusta la escalada.", False),
                ("jazz", "Me gusta escuchar jazz.", False),
                ("ceramica", "Me gusta hacer cerámica.", False),
            ),
        ),
    )

    for bank_name, query, candidates in banks:
        texts = [query] + [text for _, text, _ in candidates]
        response = client.embeddings.create(model=model, input=texts)
        vectors = [item.embedding for item in response.data]
        qvec = vectors[0]
        scored = []
        for (name, text, relevant), vector in zip(candidates, vectors[1:]):
            scored.append((cosine(qvec, vector), name, relevant, text))
        scored.sort(reverse=True)
        print(f"BANK={bank_name}")
        for rank, (score, name, relevant, text) in enumerate(scored, 1):
            print(f"{rank}\t{score:.6f}\t{'R' if relevant else 'D'}\t{name}\t{text}")
        print()


if __name__ == "__main__":
    main()
