"""Tests for the M1 harness (observatory/m1_fractal_memory.py), required by the protocol before DEV runs."""
import copy
import inspect
import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("networkx")

from observatory import m1_fractal_memory as m1  # noqa: E402


def planted(clusters=3, per=8, dims=16, seed=0):
    """Turns in planted clusters, two sessions, with texts that share a rare word inside each cluster."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(clusters, dims))
    vectors, texts, session_of, truth = [], {}, {}, []
    for c in range(clusters):
        for k in range(per):
            v = centres[c] + 0.15 * rng.normal(size=dims)
            vectors.append(v / np.linalg.norm(v))
            i = len(truth)
            texts[i] = f"Ana (1 May, 2023): turn {i} about topic{c} and common words here"
            session_of[i] = 1 + (k % 2)
            truth.append(c)
    return np.array(vectors, dtype=np.float32), texts, session_of, truth


def conversation(n_sessions=6, per_session=5, name="conv-x"):
    turns, number = [], 0
    for s in range(1, n_sessions + 1):
        for k in range(per_session):
            turns.append(m1.Turn(number, f"D{s}:{k + 1}", s, f"1:00 pm on {s} May, 2023", f"2023-05-{s:02d}",
                                 f"Ana (1:00 pm on {s} May, 2023): I planted tomato number {number} in {name}"))
            number += 1
    return m1.Conversation(name, list(range(1, n_sessions + 1)), turns, [])


def test_ari_known_values():
    assert m1.ari([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert m1.ari([0, 0, 1, 1], [0, 1, 0, 1]) <= 0.0


def test_discovery_is_deterministic_and_finds_planted_groups():
    vectors, texts, session_of, truth = planted()
    turns = list(range(len(truth)))
    a = m1.discover(vectors, turns, texts, session_of, (1, 2, 3, 4, 5))
    b = m1.discover(vectors, turns, texts, session_of, (1, 2, 3, 4, 5))
    assert json.dumps(a.to_json(), sort_keys=True) == json.dumps(b.to_json(), sort_keys=True)
    coarsest = [g.members for g in a.groups.values() if g.level == 0]
    assert coarsest, "a stable level should exist on planted clusters"
    for members in coarsest:
        assert len({truth[m] for m in members}) == 1


def test_no_stable_structure_falls_back_to_flat():
    vectors = np.eye(6, dtype=np.float32)  # nothing similar to anything
    texts = {i: f"x{i}" for i in range(6)}
    s = m1.discover(vectors, list(range(6)), texts, {i: i for i in range(6)}, (1, 2, 3, 4, 5))
    content = {i: f"turn {i}" for i in range(6)}
    sims = np.arange(6, dtype=np.float32)
    assert m1.packet(s, sims, content, 100).turns == m1.packet(None, sims, content, 100).turns


def test_packets_respect_budget_and_never_repeat_a_turn():
    vectors, texts, session_of, truth = planted()
    s = m1.discover(vectors, list(range(len(truth))), texts, session_of, (1, 2, 3, 4, 5))
    content = {i: t for i, t in texts.items()}
    for budget in (20, 64, 200, 10_000):
        for structure in (None, s):
            p = m1.packet(structure, vectors @ vectors[3], content, budget)
            assert len(p.turns) == len(set(p.turns))
            assert p.tokens <= budget
            assert len("\n".join(content[i] for i in p.turns)) <= 4 * budget


def test_walk_ends_on_cyclic_links_and_respects_the_pop_bound():
    s = m1.Structure()
    for k in range(60):
        s.groups[f"g{k}"] = m1.Group(f"g{k}", 0, {k})
        s.roots.append(f"g{k}")
    for k in range(60):  # a ring of links, plus self-referencing neighbours
        s.groups[f"g{k}"].links |= {f"g{(k + 1) % 60}", f"g{(k - 1) % 60}"}
    content = {k: f"turn {k}" for k in range(60)}
    p = m1.packet(s, np.linspace(0, 1, 60), content, 10_000)
    assert len(p.visited) <= m1.LEAF_GROUPS  # leaves only: stops after 4 groups
    for k in range(60):
        s.groups[f"g{k}"].children = [f"g{(k + 1) % 60}"]  # a cycle of children
    p = m1.packet(s, np.linspace(0, 1, 60), content, 10_000)
    assert len(p.visited) <= m1.MAX_POPS


def test_forgetting_reaches_members_centroids_and_links():
    vectors, texts, session_of, truth = planted()
    turns = list(range(len(truth)))
    s = m1.discover(vectors, turns, texts, session_of, (1, 2, 3, 4, 5))
    gone = {0, 1, 2, 8}
    left = set(turns) - gone
    m1.forget_in(s, gone, vectors, left, texts, session_of)
    for gid, group in s.groups.items():
        assert not group.members & gone
        assert len(group.members) >= 2
        mean = vectors[sorted(group.members)].mean(axis=0)
        assert np.allclose(mean / np.linalg.norm(mean), s.centroids[gid], atol=1e-6)
    assert not s.loose & gone
    again = copy.deepcopy(s)
    assert not m1.relink(again, m1.signal_graph(vectors, sorted(left), texts, session_of))


def test_retiring_the_last_members_leaves_no_empty_group():
    vectors, texts, session_of, truth = planted(clusters=2, per=4)
    s = m1.discover(vectors, list(range(8)), texts, session_of, (1, 2, 3, 4, 5))
    victim = next(iter(s.groups.values()))
    gone = set(victim.members)
    m1.forget_in(s, gone, vectors, set(range(8)) - gone, texts, session_of)
    assert all(len(g.members) >= 2 and not g.members & gone for g in s.groups.values())


def test_revision_is_bounded_versioned_and_reversible():
    vectors, texts, session_of, truth = planted(clusters=3, per=10)
    history = [i for i in range(30) if i % 10 < 6]
    s = m1.discover(vectors, history, texts, session_of, (1, 2, 3, 4, 5))
    saved = s.to_json()
    for i in range(30):
        if i not in history:
            m1.attach(s, i, vectors)
    ops = m1.revise(s, vectors, set(range(30)), texts, session_of, (1, 2, 3, 4, 5), ["D9:1"])
    assert 0 <= ops <= m1.MAX_OPS
    assert s.version == 1
    assert all(r["trigger"] == ["D9:1"] for r in s.revisions)
    content = {i: texts[i] for i in range(30)}
    reloaded = m1.Structure.from_json(json.loads(json.dumps(s.to_json())))
    sims = vectors @ vectors[5]
    assert m1.packet(s, sims, content, 200).turns == m1.packet(reloaded, sims, content, 200).turns
    first = m1.Structure.from_json(saved)  # the version before revising is still there to roll back to
    assert first.version == 0


def test_shuffled_control_keeps_the_shape():
    vectors, texts, session_of, truth = planted()
    s = m1.discover(vectors, list(range(len(truth))), texts, session_of, (1, 2, 3, 4, 5))
    x = m1.shuffled(s, vectors, seed=7)
    assert set(s.groups) == set(x.groups)
    for gid in s.groups:
        assert len(s.groups[gid].members) == len(x.groups[gid].members)
        assert s.groups[gid].children == x.groups[gid].children
    if any(len(g.members) < len(truth) for g in s.groups.values()):
        assert any(s.groups[g].members != x.groups[g].members for g in s.groups)


def test_time_tree_is_blocks_of_sessions():
    conv = conversation(n_sessions=6)
    tree = m1.time_tree(conv, {t.index for t in conv.turns})
    assert tree.roots == ["B0", "B1"]
    assert tree.groups["B0"].children == ["S1", "S2", "S3", "S4"]
    assert tree.groups["B1"].members == {t.index for t in conv.turns if t.session >= 5}


def test_cutoff_is_sixty_percent_of_sessions():
    assert conversation(n_sessions=19).cutoff == 12
    assert conversation(n_sessions=31).cutoff == 19


def test_builders_never_receive_a_question():
    forbidden = {"question", "questions", "answer", "answers", "evidence", "qa", "query", "q", "qvec"}
    for function in m1.BUILDERS:
        assert not forbidden & set(inspect.signature(function).parameters), function.__name__


def test_confirm_stays_closed_without_a_committed_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(m1, "FREEZE", tmp_path / "missing.json")
    convs = [conversation(name=f"conv-{k}") for k in range(5)]
    with pytest.raises(SystemExit):
        m1.chosen_split(convs, "confirm")
    assert len(m1.chosen_split(convs, "dev")) == 2


def test_split_follows_the_hash_rule():
    convs = [conversation(name=f"conv-{k}") for k in range(10)]
    dev, confirm = m1.split(convs)
    ordered = sorted((c.sample_id for c in convs), key=lambda i: m1.sha(f"INFINITO/M1/v1/conv/{i}"))
    assert [c.sample_id for c in dev] == ordered[:2]
    assert [c.sample_id for c in confirm] == ordered[2:]


def test_scope_isolation_between_two_projects(tmp_path):
    first, second = conversation(name="conv-a"), conversation(name="conv-b")
    result = m1.scope_check(first, second, tmp_path)
    assert result["passed"], result


def test_paired_difference_and_interval():
    values = {"c1": [(1, 0), (1, 1)], "c2": [(1, 0), (0, 0)]}
    m = m1.paired(values)
    assert m["n"] == 4 and m["diff"] == pytest.approx(50.0)
    assert m["low"] <= m["diff"] <= m["high"]
    same = m1.paired({"c1": [(1, 1)], "c2": [(0, 0)]})
    assert same["diff"] == 0 and same["low"] == 0 and same["high"] == 0


def test_audit_agreement_counts_only_labelled_items():
    stage2 = {"results": [{"rows": [{"qid": "c/1", "arm": "A0", "correct": True},
                                    {"qid": "c/2", "arm": "A1", "correct": False}]}]}
    key = lambda qid, arm: m1.sha(f"INFINITO/M1/v1/audit/{qid}/{arm}")[:12]
    labelled = [{"key": key("c/1", "A0"), "your_verdict": "correct"},
                {"key": key("c/2", "A1"), "your_verdict": "CORRECT"},
                {"key": key("c/2", "A1"), "your_verdict": ""}]
    assert m1.audit_agreement(stage2, labelled) == {"agree": 1, "total": 2}


def test_progress_survives_a_restart(tmp_path):
    path = tmp_path / "run.progress.jsonl"
    first = m1.Progress(path)
    first.put("main|c/1|A0", {"qid": "c/1", "arm": "A0", "reply": "Sweden", "correct": True})
    again = m1.Progress(path)
    assert again.get("main|c/1|A0") == {"qid": "c/1", "arm": "A0", "reply": "Sweden", "correct": True}
    assert again.get("main|c/1|A1") is None


class HashRanker:
    """Deterministic stand-in for the embeddings model: a unit vector from each text's hash."""

    def vector(self, text):
        v = np.random.default_rng(int(m1.sha(text)[:8], 16)).normal(size=16).astype(np.float32)
        return v / np.linalg.norm(v)


def asked(name="conv-s"):
    conv = conversation(name=name)
    conv.questions = [
        m1.Question(f"{name}/0", 0, "single-hop", "Which tomato came first?", "tomato number 0", ["D1:1"]),
        m1.Question(f"{name}/1", 1, "single-hop", "Which tomato in session two?", "tomato number 5", ["D2:1"]),
        m1.Question(f"{name}/2", 2, "multi-hop", "Which tomatoes in sessions one and three?", "0 and 10",
                    ["D1:1", "D3:1"]),
        m1.Question(f"{name}/3", 3, "adversarial", "What colour is the moon?", "", []),
    ]
    return conv


class FakeOllama:
    """Records every call; optionally stops the run at the n-th judge call, as a cut connection would."""

    def __init__(self, stop_at_judgment=None):
        self.calls, self.stop_at = [], stop_at_judgment

    def __call__(self, model, prompt, max_tokens=64, **_):
        self.calls.append((model, prompt))
        if model == "judge" and self.stop_at and sum(m == "judge" for m, _ in self.calls) == self.stop_at:
            raise OSError("cut")
        return ("CORRECT" if model == "judge" else "tomato number 0"), {"seconds": 0.0, "attempt": 1}


def test_stage2_reads_every_row_before_judging_and_resumes_in_either_pass(tmp_path, monkeypatch):
    conv, ranker = asked(), HashRanker()
    progress_file = tmp_path / "run.progress.jsonl"
    stopped = FakeOllama(stop_at_judgment=5)
    monkeypatch.setattr(m1, "ollama", stopped)
    (tmp_path / "a").mkdir()
    with pytest.raises(OSError):
        m1.stage2_conversation(conv, ranker, tmp_path / "a", "reader", "judge", progress=m1.Progress(progress_file))
    models = [m for m, _ in stopped.calls]
    first_judgment = models.index("judge")
    assert first_judgment == 16 + 8 and "reader" not in models[first_judgment:]  # 4 x 4 main, 2 x 4 forgetting
    assert not any(arm in p for m, p in stopped.calls if m == "judge" for arm in m1.ARMS_STAGE2)

    resumed = FakeOllama()
    monkeypatch.setattr(m1, "ollama", resumed)
    (tmp_path / "b").mkdir()
    out = m1.stage2_conversation(conv, ranker, tmp_path / "b", "reader", "judge", progress=m1.Progress(progress_file))
    assert [m for m, _ in resumed.calls] == ["judge"] * (12 + 8 - 4)  # adversarial rows need no judge

    whole = FakeOllama()
    monkeypatch.setattr(m1, "ollama", whole)
    (tmp_path / "c").mkdir()
    assert m1.stage2_conversation(conv, ranker, tmp_path / "c", "reader", "judge") == out
    assert all(r.get("correct") is not None for r in out["rows"]) and len(out["forgetting"]) == 8
    assert all("judge" in r and "revealed" in r for r in out["forgetting"])


def test_forgetting_reaches_saved_versions_and_the_revision_log(tmp_path):
    vectors, texts, session_of, truth = planted(clusters=3, per=10)
    history = [i for i in range(30) if i % 10 < 6]
    s = m1.discover(vectors, history, texts, session_of, (1, 2, 3, 4, 5))
    versions = [s.to_json()]
    gone = set(sorted(next(iter(s.groups.values())).members)[:2])
    assert any(gone & set(g["members"]) for g in versions[0]["groups"].values())
    after = m1.forget_versions(versions, gone, vectors, set(history) - gone, texts, session_of)
    assert after[0]["version"] == versions[0]["version"]
    assert m1.leftovers(m1.Structure.from_json(after[0]), gone, vectors) == (0, 0)
    conv = asked()
    memory = m1.open_memory(tmp_path, conv)
    found = m1.forgetting_checks(conv, memory, m1.embed(HashRanker(), memory.content, len(conv.turns)), HashRanker())
    memory.agent.close()
    assert found["forgotten_turns"] == 2 and found["passed"] and found["history_passed"]
    assert found["in_revision_log"] == 0


def test_packets_report_search_work_and_their_trace():
    vectors, texts, session_of, truth = planted()
    s = m1.discover(vectors, list(range(len(truth))), texts, session_of, (1, 2, 3, 4, 5))
    content = {i: texts[i] for i in range(len(truth))}
    sims = vectors @ vectors[3]
    walked, flat = m1.packet(s, sims, content, 200), m1.packet(None, sims, content, 200)
    assert walked.visited and walked.scored > 0
    assert not flat.visited and flat.scored == 0
    conv = conversation(n_sessions=6, per_session=4)
    q = m1.Question("conv-x/1", 1, "multi-hop", "q", "a", [])
    logged = json.loads(json.dumps(m1.trace(conv, q, "A2", 1, s, walked)))
    assert logged["version"] == s.version and logged["visited"][0][0] == walked.visited[0][0]
    assert [t for t, _ in logged["turns"]] == [conv.turns[i].dia_id for i in walked.turns]
    assert all(route for _, route in logged["turns"])


def test_adversarial_and_f1_scoring():
    assert m1.is_abstention("Not in memory.")
    assert not m1.is_abstention("Sweden")
    assert m1.token_f1("She moved from Sweden", "Sweden") > 0
    assert m1.judge_verdict("correct") is True and m1.judge_verdict("WRONG.") is False
