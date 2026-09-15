from src.infinito3.context_intent import resolve_context_intent


def test_profile_query_decomposition():
    intent = resolve_context_intent("Read back my name, home city, bicycle, language and occupation.")
    assert intent.mode == "facts"
    assert intent.predicates == frozenset({"name", "location", "bike", "studying_language", "occupation"})
