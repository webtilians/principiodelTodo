from src.infinito3.benchmark_cases import extended_evaluation_suite
from src.infinito3.precision_cases import precision_evaluation_suite


def test_precision_bank_is_separate_and_has_valid_controls():
    bank = precision_evaluation_suite()
    assert len(bank) == 12
    assert len({s.name for s in bank}) == 12
    assert not {s.name for s in bank} & {s.name for s in extended_evaluation_suite()}
    assert all(s.history_limit == 0 for s in bank)
    for s in bank:
        assert not set(s.expectation.context_contains) & set(s.expectation.context_excludes)
    control = next(s for s in bank if s.name == 'arithmetic_needs_no_memory')
    assert control.expectation.context_contains == ()
    assert control.expectation.answer_contains == ('4',)
