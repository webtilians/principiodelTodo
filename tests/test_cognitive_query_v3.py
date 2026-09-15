def test_cognitive_query_module_imports():
    from src.infinito3.cognitive_query import build_cognitive_query_plan
    assert build_cognitive_query_plan("What is 2 times 3?").retrieve is False
