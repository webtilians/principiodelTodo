from src.infinito3.context_intent import detect_history_cue, resolve_context_intent


def test_no_longer_enjoy_language_normalizes_to_same_operator():
    assert detect_history_cue("Sea kayaking isn't something I enjoy anymore.") == "no_longer_enjoy"
    intent = resolve_context_intent("Which activity did I say I no longer enjoy?")
    assert intent.mode == "preferences"
    assert intent.historical is True
    assert intent.history_cue == "no_longer_enjoy"
