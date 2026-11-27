"""The R1 harness keeps DEV and CONFIRM apart and fuses rankings as its protocol says (no model needed)."""
import re

from observatory import r1_recall_hybrid as r1


def test_dev_and_confirm_share_no_question_no_code_and_no_changed_fact():
    # As the protocol states: F1, F3 and F4 are disjoint. F2 draws (client, topic) from one pool in both splits,
    # so some F2 questions repeat with other answers; the report says so (nothing was tuned on DEV).
    dev, dev_golds = r1.build("dev")
    confirm, confirm_golds = r1.build("confirm")
    assert {q.text for q in dev if q.family != "F2"}.isdisjoint(q.text for q in confirm if q.family != "F2")
    codes = lambda qs: {c for q in qs if q.family == "F1" for c in re.findall(r"\d{4}", q.text)}
    assert codes(dev) and codes(dev).isdisjoint(codes(confirm))
    pairs = lambda qs: {q.text for q in qs if q.family == "F4"}
    assert pairs(dev).isdisjoint(pairs(confirm))
    assert [len([q for q in dev if q.family == f]) for f in r1.COUNTS] == list(r1.COUNTS.values())
    # The later value of a changed fact is dated after the earlier one.
    days = {g.text: g.day for g in dev_golds}
    assert all(days[q.gold] > days[q.older] for q in dev if q.family == "F4")


def test_the_data_is_the_same_on_every_run():
    assert [q.text for q in r1.build("confirm")[0]] == [q.text for q in r1.build("confirm")[0]]
    assert [d.text for d in r1.distractors("dev", 50)] == [d.text for d in r1.distractors("dev", 50)]


def test_distractors_never_use_a_gold_code():
    golds = {c for q in r1.build("dev")[0] + r1.build("confirm")[0] for c in re.findall(r"\d{4}", q.text)}
    used = {c for d in r1.distractors("dev", 2000) for c in re.findall(r"\d{4}", d.text)}
    assert golds.isdisjoint(used)


def test_rrf_rewards_what_both_rankings_put_high_and_breaks_ties_by_the_first():
    assert r1._rrf([1, 2, 3], [3, 1, 2], tiebreak=[1, 2, 3])[0] == 1
    assert r1._rrf([5, 6], [6, 5], tiebreak=[5, 6]) == [5, 6]


def test_fts_finds_a_code_among_ordinary_notes():
    texts = [f"Nota {i} del proyecto Atlas." for i in range(50)] + ["El ticket ATL-4821 del proyecto Atlas: falla."]
    assert r1.Fts(texts).rank("¿Qué pasa con el ticket ATL-4821?", 5)[0] == 50


def test_a_deleted_word_does_not_stay_in_the_fts_file():
    assert r1.deletion_check()
