"""M1: structures the memory discovers and revises, against flat retrieval and a fixed hierarchy.

See docs/INFINITO_M1_FRACTAL_MEMORY_PROTOCOL_2026-11-20.md, merged before this code was written (PR #177). Nothing
here may change an arm, a parameter or the data after a result is seen; a harness fix that changes none of them is
logged in the report.

    python -m observatory.m1_fractal_memory stage1 --data locomo10.json --model <embeddings folder> --split dev
    python -m observatory.m1_fractal_memory stage2 --data locomo10.json --model <embeddings folder> --split dev \
        --reader qwen3.8-motor [--judge <model>] [--limit N]

No model builds structure: discovery is Louvain community detection over generic signals (meaning, sequence,
shared rare words). Models are only called in stage 2, to read packets (reader) and to compare answers (judge).
Turns go through AgentMemory.remember, the real write path, one fresh store per conversation. No personal data.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import inspect
import json
import math
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --- fixed by the protocol ----------------------------------------------------------------------------------------

KNN = 10                       # meaning signal: each turn's 10 nearest turns
RARE_PERCENTILE = 5            # shared-rare-word signal: rarest 5% of the vocabulary (words in two or more turns)
RESOLUTIONS = (0.5, 1.0, 2.0, 4.0)
REPLICA_SEEDS = {1: (1, 2, 3, 4, 5), 2: (6, 7, 8, 9, 10), 3: (11, 12, 13, 14, 15)}
ARI_MIN = 0.5                  # a resolution is kept as a level only if its 5 partitions agree this much
MAX_GROUPS_FRACTION = 1 / 3    # growth bound: a level with more than n/3 groups is dropped
NEST_MAJORITY = 0.5
NEST_LINK = 0.25
LINK_TOP = 95                  # cross-link when a pair's weight is in the top 5% of the level's pairs
LEAF_TURNS = 6
LEAF_GROUPS = 4
MAX_POPS = 50
ATTACH_PERCENTILE = 25
POOL_MIN = 6
SPLIT_GROWTH = 1.5             # a group that has grown by 50% since its last version
MAX_OPS = 10
A1_BLOCK = 4                   # A1: blocks of 4 consecutive sessions
CUTOFF_FRACTION = 0.6
BUDGETS = (512, 1024, 2048)
DECISIVE_BUDGET = 1024
CEILING_RULE = 0.90            # DEV rule: B moves to 512 if A0 already covers 90% of relational questions at 1,024
STAGE2_PER_TYPE = 8
FORGET_PER_CONV = 5
BOOTSTRAP = 5000
SEED = 20261120
NAMESPACE = "INFINITO/M1/v1"

# Checked on DEV on 2026-11-20 by reading ten questions of each number (conv-26 and conv-47): 1 asks for several
# facts gathered across sessions, 2 for a date or duration, 3 for an inference beyond what was said, 4 for one fact,
# 5 swaps the speakers so that the conversation holds no answer.
TYPES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}
RELATIONAL = ("multi-hop", "temporal")
LOCAL = ("single-hop", "open-domain")

ABSTAIN = "Not in memory"
READER_PROMPT = ("Answer the question using only the memory below. If the memory does not contain the answer, reply "
                 "exactly: Not in memory. Use at most 15 words.")
JUDGE_PROMPT = ("Question, reference answer, response. Does the response give the same answer as the reference "
                "(wording, extra detail and date format may differ)? Reply CORRECT or WRONG.")
OLLAMA = "http://localhost:11434"

ARMS_STAGE1 = ("A0", "A1", "A2", "A3", "A2-shuffled", "A2-rebuilt")
ARMS_STAGE2 = ("A0", "A1", "A2", "A3")
FREEZE = Path("docs/INFINITO_M1_FREEZE.json")


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- data -----------------------------------------------------------------------------------------------------------

@dataclass
class Turn:
    index: int          # position in the whole conversation, dialogue order
    dia_id: str
    session: int
    date: str           # as LoCoMo writes it
    day: str            # ISO date for the write path
    text: str           # what is written to memory: "<speaker> (<date>): <text> [shares a photo: ...]"


@dataclass
class Question:
    qid: str
    index: int
    type: str
    question: str
    answer: str
    evidence: List[str]


@dataclass
class Conversation:
    sample_id: str
    sessions: List[int]
    turns: List[Turn]
    questions: List[Question]

    @property
    def cutoff(self) -> int:
        """The last session of the history the structure is discovered from."""
        return self.sessions[math.ceil(CUTOFF_FRACTION * len(self.sessions)) - 1]


def _day(date: str) -> str:
    for pattern in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return datetime.strptime(date.strip(), pattern).date().isoformat()
        except ValueError:
            pass
    found = re.search(r"(\d{1,2}) (\w+),? (\d{4})", date)
    if found:
        for pattern in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(" ".join(found.groups()), pattern).date().isoformat()
            except ValueError:
                pass
    raise ValueError(f"unreadable session date: {date!r}")


def _evidence(raw: Sequence[str]) -> List[str]:
    ids: List[str] = []
    for item in raw or []:
        ids.extend(part for part in re.split(r"[;,\s]+", str(item)) if part)
    return ids


def load(path: str) -> List[Conversation]:
    conversations = []
    for sample in json.loads(Path(path).read_text(encoding="utf-8")):
        conv = sample["conversation"]
        numbers = sorted(int(k.split("_")[1]) for k in conv
                         if re.fullmatch(r"session_\d+", k) and conv[k])
        turns: List[Turn] = []
        for number in numbers:
            date = conv[f"session_{number}_date_time"]
            for raw in conv[f"session_{number}"]:
                text = f"{raw['speaker']} ({date}): {raw['text']}"
                if raw.get("blip_caption"):
                    text += f" [shares a photo: {raw['blip_caption']}]"
                turns.append(Turn(len(turns), raw["dia_id"], number, date, _day(date), text))
        questions = [Question(f"{sample['sample_id']}/{i}", i, TYPES[q["category"]], q["question"],
                              str(q.get("answer", q.get("adversarial_answer", ""))), _evidence(q.get("evidence")))
                     for i, q in enumerate(sample["qa"])]
        conversations.append(Conversation(sample["sample_id"], numbers, turns, questions))
    return conversations


def split(conversations: Sequence[Conversation]) -> Tuple[List[Conversation], List[Conversation]]:
    ordered = sorted(conversations, key=lambda c: sha(f"{NAMESPACE}/conv/{c.sample_id}"))
    return ordered[:2], ordered[2:]


# --- memory ---------------------------------------------------------------------------------------------------------

@dataclass
class Memory:
    """One store and what each turn became in it."""
    agent: object
    record_of: Dict[int, str]       # turn index -> memory id
    content: Dict[int, str]         # turn index -> the text the store keeps (what every arm shows)
    refused: Dict[int, str]         # turn index -> the write path's answer
    available: Set[int]             # turns active in the store

    def refresh(self, project: Optional[str] = None) -> None:
        active = {r.id: r.content for r in self.agent._active(project)}
        self.available = {i for i, rid in self.record_of.items() if rid in active}
        self.content = {i: active[self.record_of[i]] for i in self.available}


def write(agent, conv: Conversation, project: str, turns: Optional[Iterable[Turn]] = None) -> Memory:
    record_of, refused = {}, {}
    for turn in (conv.turns if turns is None else turns):
        answer = agent.remember(turn.text, kind="fact", source="LoCoMo", project=project, date=turn.day)
        found = re.match(r"Remembered as (\S+?)\.", answer)
        if found:
            record_of[turn.index] = found.group(1)
        else:
            refused[turn.index] = answer[:120]
    memory = Memory(agent, record_of, {}, refused, set())
    memory.refresh(project)
    return memory


def open_memory(folder: Path, conv: Conversation):
    from src.infinito3.mcp_server import AgentMemory

    agent = AgentMemory(str(folder / f"{conv.sample_id}.db"))
    return write(agent, conv, conv.sample_id)


# --- generic signals and discovery ------------------------------------------------------------------------------------

def words(text: str) -> Set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def signal_graph(vectors, turns: Sequence[int], texts: Dict[int, str], session_of: Dict[int, int]):
    """Weighted graph over turns from meaning (kNN), sequence and shared rare words. No category of any kind."""
    import networkx as nx
    import numpy as np

    graph = nx.Graph()
    graph.add_nodes_from(turns)
    if len(turns) < 2:
        return graph
    order = sorted(turns)
    sub = vectors[order]
    sims = sub @ sub.T
    np.fill_diagonal(sims, -np.inf)
    k = min(KNN, len(order) - 1)
    weight: Dict[Tuple[int, int], float] = {}
    knn: Dict[Tuple[int, int], float] = {}
    for row, i in enumerate(order):
        best = np.argsort(-sims[row], kind="stable")[:k]
        for col in best:
            s = float(sims[row, col])
            if s <= 0:
                continue
            pair = (min(i, order[col]), max(i, order[col]))
            knn[pair] = max(knn.get(pair, 0.0), s)
    for pair, s in knn.items():
        weight[pair] = s
    unit = float(np.median(list(knn.values()))) if knn else 0.0
    if unit > 0:
        for a, b in zip(order, order[1:]):
            if session_of[a] == session_of[b]:
                weight[(a, b)] = weight.get((a, b), 0.0) + unit
        holders: Dict[str, List[int]] = {}
        for i in order:
            for w in words(texts[i]):
                holders.setdefault(w, []).append(i)
        shared = {w: h for w, h in holders.items() if len(h) >= 2}  # a word in one turn cannot be shared
        if shared:
            cut = float(np.percentile([len(h) for h in shared.values()], RARE_PERCENTILE, method="lower"))
            for w, h in shared.items():
                if len(h) <= cut:
                    for x in range(len(h)):
                        for y in range(x + 1, len(h)):
                            pair = (h[x], h[y])
                            weight[pair] = weight.get(pair, 0.0) + unit
    graph.add_weighted_edges_from((a, b, w) for (a, b), w in weight.items())
    return graph


def ari(a: Sequence[int], b: Sequence[int]) -> float:
    """Adjusted Rand index of two labelings of the same items."""
    n = len(a)
    if n < 2:
        return 1.0
    pairs = lambda c: c * (c - 1) / 2
    together = sum(pairs(c) for c in Counter(zip(a, b)).values())
    in_a = sum(pairs(c) for c in Counter(a).values())
    in_b = sum(pairs(c) for c in Counter(b).values())
    expected = in_a * in_b / pairs(n)
    top = (in_a + in_b) / 2
    if top == expected:
        return 1.0
    return (together - expected) / (top - expected)


@dataclass
class Group:
    gid: str
    level: int                  # kept level it came from (0 = coarsest); -1 for groups made after the cutoff
    members: Set[int]
    children: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    links: Set[str] = field(default_factory=set)
    size_at_version: int = 0


@dataclass
class Structure:
    groups: Dict[str, Group] = field(default_factory=dict)
    roots: List[str] = field(default_factory=list)
    loose: Set[int] = field(default_factory=set)
    levels: List[dict] = field(default_factory=list)       # what discovery kept, for the report
    attach_threshold: float = 1.0
    merge_threshold: Dict[int, float] = field(default_factory=dict)
    centroids: Dict[str, object] = field(default_factory=dict)
    version: int = 0
    revisions: List[dict] = field(default_factory=list)
    counter: int = 0

    def leaves(self) -> List[str]:
        return [g for g, group in self.groups.items() if not group.children]

    def to_json(self) -> dict:
        return {"groups": {g: {"level": x.level, "members": sorted(x.members), "children": list(x.children),
                               "parent": x.parent, "links": sorted(x.links), "size_at_version": x.size_at_version}
                           for g, x in self.groups.items()},
                "roots": list(self.roots), "loose": sorted(self.loose), "levels": self.levels,
                "attach_threshold": self.attach_threshold,
                "merge_threshold": {str(k): v for k, v in self.merge_threshold.items()},
                "centroids": {g: [float(v) for v in c] for g, c in self.centroids.items()},
                "version": self.version, "counter": self.counter}

    @classmethod
    def from_json(cls, data: dict) -> "Structure":
        import numpy as np

        groups = {g: Group(g, x["level"], set(x["members"]), list(x["children"]), x["parent"], set(x["links"]),
                           x["size_at_version"]) for g, x in data["groups"].items()}
        return cls(groups, list(data["roots"]), set(data["loose"]), data["levels"], data["attach_threshold"],
                   {int(k): v for k, v in data["merge_threshold"].items()},
                   {g: np.array(c, dtype=np.float32) for g, c in data["centroids"].items()},
                   data["version"], [], data["counter"])


def _partitions(graph, turns: Sequence[int], seeds: Sequence[int]):
    import networkx as nx

    kept = []
    order = sorted(turns)
    for resolution in RESOLUTIONS:
        parts = [nx.community.louvain_communities(graph, weight="weight", resolution=resolution, seed=s)
                 for s in seeds]
        labels = []
        for part in parts:
            label = {}
            for k, community in enumerate(part):
                for node in community:
                    label[node] = k
            labels.append([label[t] for t in order])
        scores = [[ari(labels[i], labels[j]) for j in range(len(parts)) if j != i] for i in range(len(parts))]
        stability = sum(sum(s) for s in scores) / (len(parts) * (len(parts) - 1))
        medoid = max(range(len(parts)), key=lambda i: (sum(scores[i]), -i))
        groups = sorted((set(c) for c in parts[medoid] if len(c) >= 2), key=min)
        record = {"resolution": resolution, "stability": round(stability, 4), "groups": len(groups)}
        if stability < ARI_MIN:
            record["dropped"] = "unstable"
        elif len(groups) > len(order) * MAX_GROUPS_FRACTION:
            record["dropped"] = "too many groups"
        elif not groups:
            record["dropped"] = "no group of two or more"
        elif any(groups == earlier for earlier in kept):
            record["dropped"] = "same partition as another kept level"  # it would only add an identical step
        else:
            record["sets"] = groups
            kept.append(groups)
        yield record


def discover(vectors, turns: Sequence[int], texts: Dict[int, str], session_of: Dict[int, int],
             seeds: Sequence[int]) -> Structure:
    """Structure found in these turns alone. Receives no question, answer or evidence id."""
    graph = signal_graph(vectors, turns, texts, session_of)
    return assemble(graph, vectors, turns, seeds)


def assemble(graph, vectors, turns: Sequence[int], seeds: Sequence[int], prefix: str = "") -> Structure:
    import numpy as np

    records = list(_partitions(graph, turns, seeds))
    kept = sorted((r for r in records if "sets" in r), key=lambda r: (r["groups"], r["resolution"]))
    structure = Structure(levels=[{k: v for k, v in r.items() if k != "sets"} for r in records])
    level_groups: List[List[str]] = []
    for level, record in enumerate(kept):
        ids = []
        for k, members in enumerate(record["sets"]):
            gid = f"{prefix}L{level}g{k}"
            structure.groups[gid] = Group(gid, level, set(members), size_at_version=len(members))
            ids.append(gid)
        level_groups.append(ids)
    for level, ids in enumerate(level_groups):
        for gid in ids:
            group = structure.groups[gid]
            if level == 0:
                structure.roots.append(gid)
                continue
            overlaps = [(len(group.members & structure.groups[c].members), c) for c in level_groups[level - 1]]
            best, coarse = sorted(overlaps, key=lambda x: (-x[0], x[1]))[0]
            if best > NEST_MAJORITY * len(group.members):
                group.parent = coarse
                structure.groups[coarse].children.append(gid)
            else:
                structure.roots.append(gid)
                for count, c in overlaps:
                    if count >= NEST_LINK * len(group.members):
                        group.links.add(c)
                        structure.groups[c].links.add(gid)
    grouped = set().union(*(g.members for g in structure.groups.values())) if structure.groups else set()
    structure.loose = set(turns) - grouped
    relink(structure, graph)
    recentre(structure, vectors)
    if structure.groups:
        leaf_sims = [float(vectors[m] @ structure.centroids[g]) for g in structure.leaves()
                     for m in structure.groups[g].members]
        structure.attach_threshold = float(np.percentile(leaf_sims, ATTACH_PERCENTILE))
        for level in {g.level for g in structure.groups.values()}:
            ids = [g for g, x in structure.groups.items() if x.level == level]
            best = max((float(structure.centroids[a] @ structure.centroids[b])
                        for i, a in enumerate(ids) for b in ids[i + 1:]), default=1.0)
            structure.merge_threshold[level] = best
    return structure


def recentre(structure: Structure, vectors) -> None:
    import numpy as np

    structure.centroids = {}
    for gid, group in structure.groups.items():
        mean = vectors[sorted(group.members)].astype(np.float32).mean(axis=0)
        structure.centroids[gid] = mean / (np.linalg.norm(mean) + 1e-9)


def _nest_links(structure: Structure) -> Set[Tuple[str, str]]:
    """Links that come from nesting (a finer group with no majority parent), kept when same-level links change."""
    found = set()
    for gid, group in structure.groups.items():
        for other in group.links:
            if structure.groups[other].level != group.level:
                found.add((min(gid, other), max(gid, other)))
    return found


def same_level_links(structure: Structure, graph) -> Set[Tuple[str, str]]:
    import numpy as np

    links = set()
    by_level: Dict[int, List[str]] = {}
    for gid, group in structure.groups.items():
        if len(group.members) >= 2:
            by_level.setdefault(group.level, []).append(gid)
    for level, ids in by_level.items():
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        owner: Dict[int, List[str]] = {}
        for gid in ids:
            for m in structure.groups[gid].members:
                owner.setdefault(m, []).append(gid)
        total: Dict[Tuple[str, str], float] = {}
        for u, v, data in graph.edges(data=True):
            for a in owner.get(u, ()):
                for b in owner.get(v, ()):
                    if a != b:
                        pair = (min(a, b), max(a, b))
                        total[pair] = total.get(pair, 0.0) + data["weight"]
        weights = []
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                size = math.sqrt(len(structure.groups[a].members) * len(structure.groups[b].members))
                weights.append(((a, b), total.get((a, b), 0.0) / size))
        cut = float(np.percentile([w for _, w in weights], LINK_TOP))
        links |= {pair for pair, w in weights if w > cut and w > 0}
    return links


def relink(structure: Structure, graph) -> bool:
    """Recompute same-level cross-links by the protocol's rule; nesting links stay. True if anything changed."""
    nested = _nest_links(structure)
    before = {(min(g, o), max(g, o)) for g, x in structure.groups.items() for o in x.links}
    after = nested | same_level_links(structure, graph)
    for group in structure.groups.values():
        group.links = set()
    for a, b in after:
        structure.groups[a].links.add(b)
        structure.groups[b].links.add(a)
    return before != after


# --- the fixed hierarchy (A1) -----------------------------------------------------------------------------------------

def time_tree(conv: Conversation, available: Set[int]) -> Structure:
    """Conversation -> blocks of 4 consecutive sessions -> session -> turns. Imposed by the designer."""
    structure = Structure()
    sessions = [s for s in conv.sessions if any(t.index in available for t in conv.turns if t.session == s)]
    for b in range(0, len(sessions), A1_BLOCK):
        block = f"B{b // A1_BLOCK}"
        members: Set[int] = set()
        structure.groups[block] = Group(block, 0, set())
        structure.roots.append(block)
        for s in sessions[b:b + A1_BLOCK]:
            turns = {t.index for t in conv.turns if t.session == s and t.index in available}
            gid = f"S{s}"
            structure.groups[gid] = Group(gid, 1, turns, parent=block)
            structure.groups[block].children.append(gid)
            members |= turns
        structure.groups[block].members = members
    return structure


# --- the walk and the packet -------------------------------------------------------------------------------------------

@dataclass
class Packet:
    turns: List[int]                    # dialogue order
    routes: Dict[int, str]
    visited: List[Tuple[str, float]]
    tokens: int
    flat_fill: bool
    scored: int = 0                     # member similarities the walk read: its search work, beyond the budget


class Filler:
    def __init__(self, content: Dict[int, str], budget: int):
        self.content, self.budget = content, budget
        self.chosen: List[int] = []
        self.routes: Dict[int, str] = {}
        self.chars = 0

    def tokens_with(self, index: int) -> int:
        extra = len(self.content[index]) + (1 if self.chosen else 0)
        return math.ceil((self.chars + extra) / 4)

    def add(self, index: int, route: str) -> bool:
        if index in self.routes or index not in self.content:
            return False
        if self.tokens_with(index) > self.budget:
            return False
        self.chars += len(self.content[index]) + (1 if self.chosen else 0)
        self.chosen.append(index)
        self.routes[index] = route
        return True

    def full(self) -> bool:
        return not any(self.tokens_with(i) <= self.budget for i in self.content if i not in self.routes)


def flat_order(sims, available: Set[int]) -> List[int]:
    return sorted(available, key=lambda i: (-float(sims[i]), i))


def packet(structure: Optional[Structure], sims, content: Dict[int, str], budget: int) -> Packet:
    """The shared walk (best-first over groups, down to turns) and the flat fallback. A0 is structure=None."""
    fill = Filler(content, budget)
    visited: List[Tuple[str, float]] = []
    work = [0]
    if structure is not None and structure.groups:
        def score(gid: str) -> float:
            members = structure.groups[gid].members
            work[0] += len(members)
            return max((float(sims[m]) for m in members if m in content), default=-1.0)

        heap = [(-score(g), g) for g in structure.roots]
        heapq.heapify(heap)
        seen: Set[str] = set()
        leaves = 0
        while heap and len(visited) < MAX_POPS and leaves < LEAF_GROUPS:
            negative, gid = heapq.heappop(heap)
            if gid in seen:
                continue
            seen.add(gid)
            visited.append((gid, -negative))
            group = structure.groups[gid]
            for nxt in list(group.children) + sorted(group.links):
                if nxt not in seen and nxt in structure.groups:
                    heapq.heappush(heap, (-score(nxt), nxt))
            if not group.children:
                best = sorted((m for m in group.members if m in content), key=lambda m: (-float(sims[m]), m))
                for m in best[:LEAF_TURNS]:
                    fill.add(m, gid)
                leaves += 1
            if fill.full():
                break
    before = len(fill.chosen)
    for i in flat_order(sims, set(content)):
        fill.add(i, "flat")
    order = sorted(fill.chosen)
    return Packet(order, fill.routes, visited, math.ceil(fill.chars / 4), len(fill.chosen) > before, work[0])


def render(p: Packet, content: Dict[int, str]) -> str:
    return "\n".join(content[i] for i in p.turns)


# --- the stream: frozen (A2) and revised (A3) ----------------------------------------------------------------------------

def attach(structure: Structure, index: int, vectors) -> Optional[str]:
    """A new turn joins the finest group whose centroid is closest, if close enough; else the loose pool."""
    leaves = structure.leaves()
    if not leaves:
        structure.loose.add(index)
        return None
    best = max(leaves, key=lambda g: (float(vectors[index] @ structure.centroids[g]), g))
    if float(vectors[index] @ structure.centroids[best]) < structure.attach_threshold:
        structure.loose.add(index)
        return None
    gid: Optional[str] = best
    while gid is not None:
        structure.groups[gid].members.add(index)
        gid = structure.groups[gid].parent
    return best


def _new_gid(structure: Structure, kind: str) -> str:
    structure.counter += 1
    return f"{kind}{structure.counter}"


def _retire(structure: Structure, gid: str) -> None:
    group = structure.groups.pop(gid)
    if group.parent and group.parent in structure.groups:
        structure.groups[group.parent].children.remove(gid)  # its turns are already in the parent
    elif not group.parent:
        elsewhere = set().union(*(g.members for g in structure.groups.values())) if structure.groups else set()
        structure.loose |= group.members - elsewhere
    if gid in structure.roots:
        structure.roots.remove(gid)
    for child in group.children:
        if child in structure.groups:
            structure.groups[child].parent = group.parent
            (structure.groups[group.parent].children.append(child) if group.parent in structure.groups
             else structure.roots.append(child))
    for other in group.links:
        if other in structure.groups:
            structure.groups[other].links.discard(gid)
    structure.centroids.pop(gid, None)


def _subgroups(vectors, members: Set[int], texts, session_of, seeds) -> List[Set[int]]:
    sub = discover(vectors, sorted(members), texts, session_of, seeds)
    coarsest = [g for g in sub.groups.values() if g.level == 0]
    return [g.members for g in sorted(coarsest, key=lambda g: min(g.members))]


def revise(structure: Structure, vectors, available: Set[int], texts, session_of, seeds, trigger: List[str]) -> int:
    """A3's operations after one session, at most MAX_OPS, each recorded as a revision. Sources never change."""
    ops = 0

    def log(operation: str, nodes: List[str]) -> None:
        nonlocal ops
        ops += 1
        structure.revisions.append({"version": structure.version + 1, "operation": operation, "nodes": nodes,
                                    "trigger": trigger})

    recentre(structure, vectors)
    if len(structure.loose) >= POOL_MIN and ops < MAX_OPS:
        for members in _subgroups(vectors, structure.loose, texts, session_of, seeds):
            if ops >= MAX_OPS:
                break
            gid = _new_gid(structure, "N")
            structure.groups[gid] = Group(gid, -1, set(members), size_at_version=len(members))
            structure.roots.append(gid)
            structure.loose -= members
            log("create", [gid])
    for gid in sorted(structure.leaves()):
        if ops >= MAX_OPS:
            break
        group = structure.groups[gid]
        if len(group.members) >= SPLIT_GROWTH * max(group.size_at_version, 1):
            parts = _subgroups(vectors, group.members, texts, session_of, seeds)
            group.size_at_version = len(group.members)
            if len(parts) >= 2:
                kids = []
                for members in parts:
                    child = _new_gid(structure, "P")
                    structure.groups[child] = Group(child, group.level, set(members), parent=gid,
                                                    size_at_version=len(members))
                    group.children.append(child)
                    kids.append(child)
                log("split", [gid] + kids)
    recentre(structure, vectors)
    used: Set[str] = set()
    levels = sorted({g.level for g in structure.groups.values()})
    for level in levels:
        threshold = (structure.merge_threshold.get(level) if level >= 0
                     else max(structure.merge_threshold.values(), default=1.0))
        ids = sorted(g for g, x in structure.groups.items() if x.level == level)
        pairs = sorted(((float(structure.centroids[a] @ structure.centroids[b]), a, b)
                        for i, a in enumerate(ids) for b in ids[i + 1:]), reverse=True)
        for s, a, b in pairs:
            if ops >= MAX_OPS:
                break
            if s <= threshold or a in used or b in used or a not in structure.groups or b not in structure.groups:
                continue
            ga, gb = structure.groups[a], structure.groups[b]
            gid = _new_gid(structure, "M")
            parent = ga.parent if ga.parent == gb.parent else None
            merged = Group(gid, level, ga.members | gb.members, parent=parent,
                           links=(ga.links | gb.links) - {a, b}, size_at_version=len(ga.members | gb.members))
            structure.groups[gid] = merged
            for child in ga.children + gb.children:
                structure.groups[child].parent = gid
                merged.children.append(child)
            ga.children, gb.children = [], []
            for old in (a, b):
                old_group = structure.groups[old]
                old_group.members = set()
                if old_group.parent and old_group.parent in structure.groups:
                    structure.groups[old_group.parent].children.remove(old)
                old_group.parent = None
                if old in structure.roots:
                    structure.roots.remove(old)
                for other in old_group.links:
                    if other in structure.groups:
                        structure.groups[other].links.discard(old)
                old_group.links = set()
                del structure.groups[old]
                structure.centroids.pop(old, None)
            for other in merged.links:
                if other in structure.groups:
                    structure.groups[other].links.add(gid)
            if parent and parent in structure.groups:
                structure.groups[parent].children.append(gid)
            else:
                structure.roots.append(gid)
            used |= {a, b, gid}
            log("merge", [a, b, gid])
    for gid in sorted(g for g, x in structure.groups.items() if len(x.members) < 2):
        if ops >= MAX_OPS:
            break
        _retire(structure, gid)
        log("retire", [gid])
    recentre(structure, vectors)
    if ops < MAX_OPS:
        graph = signal_graph(vectors, sorted(available), texts, session_of)
        if relink(structure, graph):
            log("relink", [])
    structure.version += 1
    return ops


def stream(conv: Conversation, memory: Memory, vectors, seeds: Sequence[int]):
    """A2 (frozen at the cutoff, later turns attached) and A3 (revised after each later session), plus A3's versions."""
    session_of = {t.index: t.session for t in conv.turns}
    history = sorted(i for i in memory.available if session_of[i] <= conv.cutoff)
    base = discover(vectors, history, memory.content, session_of, seeds)
    frozen, revised = copy.deepcopy(base), copy.deepcopy(base)
    versions = [revised.to_json()]
    for s in [s for s in conv.sessions if s > conv.cutoff]:
        new = sorted(i for i in memory.available if session_of[i] == s)
        for i in new:
            attach(frozen, i, vectors)       # frozen centroids: A2 never changes its structure
            attach(revised, i, vectors)
        seen = sorted(i for i in memory.available if session_of[i] <= s)
        revise(revised, vectors, set(seen), memory.content, session_of, seeds,
               [t.dia_id for t in conv.turns if t.session == s and t.index in memory.available])
        versions.append(revised.to_json())
    return base, frozen, revised, versions


def shuffled(structure: Structure, vectors, seed: int = SEED) -> Structure:
    """A2's exact shape, sizes and nesting with the grouped turns randomly permuted (the negative control)."""
    out = copy.deepcopy(structure)
    grouped = sorted(set().union(*(g.members for g in out.groups.values()))) if out.groups else []
    target = list(grouped)
    random.Random(seed).shuffle(target)
    mapping = dict(zip(grouped, target))
    for group in out.groups.values():
        group.members = {mapping[m] for m in group.members}
    recentre(out, vectors)
    return out


def forget_in(structure: Structure, gone: Set[int], vectors, available: Set[int], texts, session_of) -> None:
    """Forgetting reaches every structure: members, loose pool, centroids and links; groups under 2 retire."""
    for group in structure.groups.values():
        group.members -= gone
    structure.loose -= gone
    for gid in sorted(g for g, x in structure.groups.items() if len(x.members) < 2):
        if gid in structure.groups:
            _retire(structure, gid)
    structure.loose -= gone
    recentre(structure, vectors)
    relink(structure, signal_graph(vectors, sorted(available), texts, session_of))


# --- stage 1 ----------------------------------------------------------------------------------------------------------

def embed(ranker, texts: Dict[int, str], size: int):
    import numpy as np

    vectors = None
    for i, text in texts.items():
        v = ranker.vector(text)
        if vectors is None:
            vectors = np.zeros((size, v.shape[0]), dtype=np.float32)
        vectors[i] = v
    return vectors


def answerable(q: Question, by_dia: Dict[str, Turn]) -> bool:
    return bool(q.evidence) and all(e in by_dia for e in q.evidence)


def arm_structures(conv: Conversation, memory: Memory, vectors, replica: int) -> Dict[str, Optional[Structure]]:
    seeds = REPLICA_SEEDS[replica]
    session_of = {t.index: t.session for t in conv.turns}
    seconds: Dict[str, float] = {}

    def timed(name: str, build):
        started = time.perf_counter()
        made = build()
        seconds[name] = round(time.perf_counter() - started, 3)
        return made

    base, frozen, revised, versions = timed("A2+A3 stream", lambda: stream(conv, memory, vectors, seeds))
    rebuilt = timed("A2-rebuilt", lambda: discover(vectors, sorted(memory.available), memory.content, session_of,
                                                   seeds))
    tree = timed("A1", lambda: time_tree(conv, memory.available))
    control = timed("A2-shuffled", lambda: shuffled(frozen, vectors))
    return {"A0": None, "A1": tree, "A2": frozen, "A3": revised, "A2-shuffled": control, "A2-rebuilt": rebuilt,
            "_base": base, "_versions": versions, "_seconds": seconds}


def describe(structure: Optional[Structure]) -> dict:
    if structure is None:
        return {}
    links = sum(len(g.links) for g in structure.groups.values()) // 2
    ops = Counter(r["operation"] for r in structure.revisions)
    return {"groups": len(structure.groups), "roots": len(structure.roots), "leaves": len(structure.leaves()),
            "links": links, "loose": len(structure.loose), "levels": structure.levels, "version": structure.version,
            "operations": dict(ops), "bytes": len(json.dumps(structure.to_json()))}


def trace(conv: Conversation, q: Question, arm: str, replica: int, structure: Optional[Structure], p: Packet) -> dict:
    """The evidence packet as the protocol logs it: version, nodes visited with scores, and each turn's route."""
    return {"qid": q.qid, "arm": arm, "replica": replica,
            "version": None if structure is None else structure.version,
            "visited": [[gid, round(s, 4)] for gid, s in p.visited],
            "turns": [[conv.turns[i].dia_id, p.routes.get(i)] for i in p.turns],
            "tokens": p.tokens, "flat_fill": p.flat_fill, "scored": p.scored}


def stage1_conversation(conv: Conversation, ranker, folder: Path, replicas: Sequence[int], traces=None) -> dict:
    """Coverage for every answerable question, arm, budget and replica, with each packet's search cost; the full
    trace at the decisive budget goes to `traces` (a text file, one JSON line per packet) when given."""
    t0 = time.time()
    memory = open_memory(folder, conv)
    by_dia = {t.dia_id: t for t in conv.turns}
    vectors = embed(ranker, memory.content, len(conv.turns))
    built = time.time() - t0
    out = {"sample_id": conv.sample_id, "sessions": len(conv.sessions), "cutoff": conv.cutoff,
           "turns": len(conv.turns), "written": len(memory.record_of), "active": len(memory.available),
           "refused": {conv.turns[i].dia_id: why for i, why in memory.refused.items()},
           "inactive_after_write": sorted(conv.turns[i].dia_id for i in memory.record_of if i not in memory.available),
           "build_seconds": round(built, 1), "replicas": {}, "questions": []}
    excluded = [q.qid for q in conv.questions if q.evidence and not answerable(q, by_dia)]
    out["excluded_unknown_evidence"] = excluded
    lost = Counter()
    for q in conv.questions:
        if answerable(q, by_dia) and any(by_dia[e].index not in memory.available for e in q.evidence):
            lost[q.type] += 1
    out["questions_with_evidence_missing_from_store"] = dict(lost)
    qs = [q for q in conv.questions if answerable(q, by_dia)]
    qvec = {q.qid: ranker.vector(q.question) for q in qs}
    for replica in replicas:
        t1 = time.time()
        arms = arm_structures(conv, memory, vectors, replica)
        rep = {"seconds": round(time.time() - t1, 1), "build_seconds": arms["_seconds"],
               "structures": {a: describe(s) for a, s in arms.items() if not a.startswith("_")},
               "rollback": rollback_check([(arms["_base"], arms["_versions"][0]), (arms["A3"], arms["_versions"][-1])],
                                          vectors, memory.content, qs, qvec)}
        rows = []
        for q in qs:
            sims = vectors @ qvec[q.qid]
            evidence = {by_dia[e].index for e in q.evidence}
            post = any(by_dia[e].session > conv.cutoff for e in q.evidence)
            row = {"qid": q.qid, "type": q.type, "post_cutoff": post, "arms": {}}
            for arm in ARMS_STAGE1:
                if replica != replicas[0] and arm in ("A0", "A1"):
                    continue  # deterministic: computed once
                row["arms"][arm] = {}
                for budget in BUDGETS:
                    started = time.perf_counter()
                    p = packet(arms[arm], sims, memory.content, budget)
                    ms = 1000 * (time.perf_counter() - started)
                    got = evidence & set(p.turns)
                    row["arms"][arm][str(budget)] = {"full": got == evidence, "partial": len(got) / len(evidence),
                                                     "tokens": p.tokens, "flat_fill": p.flat_fill,
                                                     "visited": len(p.visited), "scored": p.scored,
                                                     "ms": round(ms, 3)}
                    if traces is not None and budget == DECISIVE_BUDGET:
                        traces.write(json.dumps(trace(conv, q, arm, replica, arms[arm], p)) + "\n")
            rows.append(row)
        rep["rows"] = rows
        out["replicas"][str(replica)] = rep
    memory.agent.close()
    (folder / "forget").mkdir(exist_ok=True)
    copy_memory = open_memory(folder / "forget", conv)
    out["forgetting"] = forgetting_checks(conv, copy_memory, vectors, ranker)
    copy_memory.agent.close()
    out["seconds"] = round(time.time() - t0, 1)
    return out


def rollback_check(pairs: Sequence[Tuple[Structure, dict]], vectors, content, questions: Sequence[Question],
                   qvec) -> bool:
    """Reloading a saved version reproduces the packets the live structure gave (A3 runs in shadow)."""
    for live, saved in pairs:
        reloaded = Structure.from_json(json.loads(json.dumps(saved)))
        for q in questions:
            sims = vectors @ qvec[q.qid]
            if packet(live, sims, content, DECISIVE_BUDGET).turns != \
                    packet(reloaded, sims, content, DECISIVE_BUDGET).turns:
                return False
    return True


def forget_questions(conv: Conversation) -> List[Question]:
    by_dia = {t.dia_id: t for t in conv.turns}
    pool = [q for q in conv.questions if q.type == "single-hop" and len(q.evidence) == 1 and answerable(q, by_dia)]
    return sorted(pool, key=lambda q: sha(f"{NAMESPACE}/forget/{conv.sample_id}/{q.index}"))[:FORGET_PER_CONV]


def forget_versions(versions: Sequence[dict], gone: Set[int], vectors, available: Set[int], texts,
                    session_of) -> List[dict]:
    """Forgetting reaches A3's saved versions too, so rolling back can never bring a forgotten turn back. Each
    version keeps its number; its members, centroids and links are rewritten as forget_in does."""
    out = []
    for saved in versions:
        old = Structure.from_json(json.loads(json.dumps(saved)))
        forget_in(old, gone, vectors, available, texts, session_of)
        out.append(old.to_json())
    return out


def forgotten_state(conv: Conversation, memory: Memory, vectors):
    """Forget the evidence of the forgetting questions through the real path; every structure follows, and so does
    A3's history: its saved versions and the turn ids in its revision log."""
    session_of = {t.index: t.session for t in conv.turns}
    arms = arm_structures(conv, memory, vectors, 1)
    by_dia = {t.dia_id: t for t in conv.turns}
    gone = {by_dia[q.evidence[0]].index for q in forget_questions(conv)} & set(memory.record_of)
    memory.agent.forget(sorted({memory.record_of[i] for i in gone}))
    memory.refresh(conv.sample_id)
    gone |= {i for i in memory.record_of if i not in memory.available}  # turns that shared a forgotten record
    state = {"A0": None, "A1": time_tree(conv, memory.available)}
    for arm in ("A2", "A3"):
        forget_in(arms[arm], gone, vectors, memory.available, memory.content, session_of)
        state[arm] = arms[arm]
    versions = forget_versions(arms["_versions"], gone, vectors, memory.available, memory.content, session_of)
    gone_ids = {conv.turns[i].dia_id for i in gone}
    for record in state["A3"].revisions:
        record["trigger"] = [d for d in record["trigger"] if d not in gone_ids]
    return state, gone, versions


def leftovers(structure: Structure, gone: Set[int], vectors) -> Tuple[int, int]:
    """Forgotten turns still listed in a structure, and centroids that are not the mean of current members."""
    import numpy as np

    listed = sum(len(g.members & gone) for g in structure.groups.values()) + len(structure.loose & gone)
    stale = 0
    for gid, group in structure.groups.items():
        if gid in structure.centroids:
            mean = vectors[sorted(group.members)].mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-9)
            stale += int(not np.allclose(mean, structure.centroids[gid], atol=1e-6))
    return listed, stale


def forgetting_checks(conv: Conversation, memory: Memory, vectors, ranker) -> dict:
    session_of = {t.index: t.session for t in conv.turns}
    state, gone, versions = forgotten_state(conv, memory, vectors)
    found = {"forgotten_turns": len(gone), "in_packet": 0, "in_group": 0, "stale_centroid": 0, "stale_links": 0}
    for arm, structure in state.items():
        if structure is not None:
            listed, stale = leftovers(structure, gone, vectors)
            found["in_group"] += listed
            found["stale_centroid"] += stale
            if arm in ("A2", "A3"):
                again = copy.deepcopy(structure)
                graph = signal_graph(vectors, sorted(memory.available), memory.content, session_of)
                found["stale_links"] += int(relink(again, graph))
        for q in conv.questions:
            p = packet(structure, vectors @ ranker.vector(q.question), memory.content, DECISIVE_BUDGET)
            found["in_packet"] += len(set(p.turns) & gone)
    found["passed"] = not (found["in_packet"] or found["in_group"] or found["stale_centroid"] or found["stale_links"])
    # Added after DEV, reported beside the four protocol checks and outside their gate: A3's history.
    found["versions"] = len(versions)
    found["in_version"] = found["stale_centroid_in_version"] = 0
    for saved in versions:
        listed, stale = leftovers(Structure.from_json(saved), gone, vectors)
        found["in_version"] += listed
        found["stale_centroid_in_version"] += stale
    gone_ids = {conv.turns[i].dia_id for i in gone}
    found["in_revision_log"] = sum(d in gone_ids for r in state["A3"].revisions for d in r["trigger"])
    found["history_passed"] = not (found["in_version"] or found["stale_centroid_in_version"]
                                   or found["in_revision_log"])
    return found


def scope_check(first: Conversation, second: Conversation, folder: Path) -> dict:
    """Two conversations in one store under two project tags: neither may reach the other."""
    from src.infinito3.mcp_server import AgentMemory

    agent = AgentMemory(str(folder / f"scope-{first.sample_id}-{second.sample_id}.db"))
    a, b = write(agent, first, first.sample_id), write(agent, second, second.sample_id)
    ids_a, ids_b = set(a.record_of.values()), set(b.record_of.values())
    leaks = len({r.id for r in agent._active(first.sample_id)} & ids_b)
    leaks += len({r.id for r in agent._active(second.sample_id)} & ids_a)
    text_b = set(b.content.values())
    for q in first.questions[:20]:
        shown = agent.recall(q.question, project=first.sample_id).split("\n")[1:]
        leaks += sum(any(line.endswith(t) or t in line for t in text_b) for line in shown)
    agent.close()
    return {"pair": [first.sample_id, second.sample_id], "leaks": leaks, "passed": leaks == 0}


def summarize(results: Sequence[dict], budget: int) -> dict:
    """Full-evidence coverage by arm and question type (replica means for discovered arms)."""
    table: Dict[str, Dict[str, List[float]]] = {}
    for conv in results:
        for replica, rep in conv["replicas"].items():
            for row in rep["rows"]:
                for arm, by_budget in row["arms"].items():
                    key = f"{arm}" if arm in ("A0", "A1") else f"{arm}#r{replica}"
                    for group in (row["type"], "relational" if row["type"] in RELATIONAL else "local",
                                  ("post-" if row["post_cutoff"] else "pre-") + (
                                      "relational" if row["type"] in RELATIONAL else "local")):
                        table.setdefault(key, {}).setdefault(group, []).append(by_budget[str(budget)]["full"])
    return {key: {g: {"n": len(v), "coverage": round(100 * sum(v) / len(v), 1)} for g, v in sorted(groups.items())}
            for key, groups in sorted(table.items())}


# --- stage 2 ----------------------------------------------------------------------------------------------------------

def ollama(model: str, prompt: str, max_tokens: int = 64, timeout: int = 300, attempts: int = 3) -> Tuple[str, dict]:
    """One deterministic call (temperature 0, fixed seed). A call that hangs, as after the laptop suspended on
    2026-11-20, is retried; the same request gives the same answer, so a retry changes nothing measured."""
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "think": False,
            "options": {"temperature": 0, "seed": SEED, "num_ctx": 8192, "num_predict": max_tokens}}
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(f"{OLLAMA}/api/chat", data=json.dumps(body).encode("utf-8"),
                                         headers={"Content-Type": "application/json"})
        started = time.time()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except (OSError, ValueError) as error:  # URLError, timeouts and cut connections are OSErrors
            if attempt == attempts:
                raise
            print(f"  retry {attempt} after {type(error).__name__}", file=sys.stderr, flush=True)
            time.sleep(10)
    meta = {"seconds": round(time.time() - started, 2), "prompt_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"), "attempt": attempt}
    return data["message"]["content"].strip(), meta


def stay_awake() -> None:
    """Ask Windows not to idle-sleep while this process runs (released when it exits). No setting is changed;
    a closed lid, a manual sleep or a critical battery still sleep."""
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED


class Progress:
    """Rows already answered, kept in a JSON-lines file next to the output, so a stopped run resumes where it was."""

    def __init__(self, path: Optional[Path]):
        self.path = path
        self.done: Dict[str, dict] = {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.done[row["_key"]] = row

    def get(self, key: str) -> Optional[dict]:
        row = self.done.get(key)
        return None if row is None else {k: v for k, v in row.items() if k != "_key"}

    def put(self, key: str, row: dict) -> None:
        self.done[key] = dict(row, _key=key)
        if self.path:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(dict(row, _key=key), ensure_ascii=False) + "\n")


def model_digest(model: str) -> Optional[str]:
    with urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=30) as response:
        for m in json.loads(response.read().decode("utf-8"))["models"]:
            if m["name"] == model or m["name"] == f"{model}:latest":
                return m["digest"]
    return None


def stage2_sample(conv: Conversation, per_type: int = STAGE2_PER_TYPE) -> List[Question]:
    chosen = []
    for kind in TYPES.values():
        pool = sorted((q for q in conv.questions if q.type == kind),
                      key=lambda q: sha(f"{NAMESPACE}/q/{conv.sample_id}/{q.index}"))
        chosen.extend(pool[:per_type])
    return chosen


def is_abstention(reply: str) -> bool:
    return reply.strip().strip(".").strip().casefold() == ABSTAIN.casefold()


def token_f1(reply: str, reference: str) -> float:
    norm = lambda s: [w for w in re.findall(r"\w+", s.casefold()) if w not in {"a", "an", "the"}]
    a, b = norm(reply), norm(reference)
    common = sum((Counter(a) & Counter(b)).values())
    if not a or not b or not common:
        return 0.0
    precision, recall = common / len(a), common / len(b)
    return 2 * precision * recall / (precision + recall)


def judge_verdict(text: str) -> Optional[bool]:
    found = re.search(r"\b(CORRECT|WRONG)\b", text.upper())
    return None if not found else found.group(1) == "CORRECT"


def judge_prompt(q: Question, reply: str) -> str:
    return f"{JUDGE_PROMPT}\n\nQuestion: {q.question}\nReference answer: {q.answer}\nResponse: {reply}"


def owes_judgment(row: dict, judge: Optional[str]) -> bool:
    """A row the reader has answered and the judge has not graded yet (adversarial rows need no model)."""
    return bool(judge) and row.get("type") != "adversarial" and "judge" not in row


def stage2_conversation(conv: Conversation, ranker, folder: Path, reader: str, judge: Optional[str],
                        limit: Optional[int] = None, progress: Optional[Progress] = None) -> dict:
    """The reader answers every row of the conversation (main, then forgetting) before the judge grades any.
    Alternating them made Ollama reload a model on every call. Each row is saved when read and again when judged,
    so a stopped run resumes in either pass; the later line of a key wins."""
    progress = progress or Progress(None)
    memory = open_memory(folder, conv)
    vectors = embed(ranker, memory.content, len(conv.turns))
    arms = arm_structures(conv, memory, vectors, 1)
    sample = stage2_sample(conv)[:limit] if limit else stage2_sample(conv)
    order = random.Random(sha(conv.sample_id)).sample(range(len(sample) * len(ARMS_STAGE2)),
                                                      len(sample) * len(ARMS_STAGE2))
    jobs = [(sample[k // len(ARMS_STAGE2)], ARMS_STAGE2[k % len(ARMS_STAGE2)]) for k in order]  # arms shuffled
    main: List[Tuple[str, Question, dict]] = []
    for q, arm in jobs:
        key = f"main|{q.qid}|{arm}"
        row = progress.get(key)
        if row is None:
            sims = vectors @ ranker.vector(q.question)
            p = packet(arms[arm], sims, memory.content, DECISIVE_BUDGET)
            prompt = f"{READER_PROMPT}\n\nMemory:\n{render(p, memory.content)}\n\nQuestion: {q.question}"
            reply, meta = ollama(reader, prompt)
            row = {"qid": q.qid, "type": q.type, "arm": arm, "reply": reply, "reader": meta, "tokens": p.tokens}
            if q.type == "adversarial":
                row["correct"] = is_abstention(reply)
            else:
                row["f1"] = round(token_f1(reply, q.answer), 4)
            progress.put(key, row)
            print(f"  {q.qid} {arm} {meta['seconds']}s", file=sys.stderr, flush=True)
        main.append((key, q, row))
    memory.agent.close()
    forgetting: List[Tuple[str, Question, dict]] = []
    if judge and not limit:
        (folder / "forget").mkdir(exist_ok=True)
        copy_memory = open_memory(folder / "forget", conv)
        state, gone, _ = forgotten_state(conv, copy_memory, vectors)
        for q in forget_questions(conv):
            for arm in ARMS_STAGE2:
                key = f"forget|{q.qid}|{arm}"
                row = progress.get(key)
                if row is None:
                    p = packet(state[arm], vectors @ ranker.vector(q.question), copy_memory.content,
                               DECISIVE_BUDGET)
                    prompt = (f"{READER_PROMPT}\n\nMemory:\n{render(p, copy_memory.content)}\n\n"
                              f"Question: {q.question}")
                    reply, meta = ollama(reader, prompt)
                    row = {"qid": q.qid, "arm": arm, "reply": reply, "forgotten_in_packet": len(set(p.turns) & gone)}
                    progress.put(key, row)
                    print(f"  forget {q.qid} {arm} {meta['seconds']}s", file=sys.stderr, flush=True)
                forgetting.append((key, q, row))
        copy_memory.agent.close()
    for key, q, row in main + forgetting:  # the judge, in the same order, never told the arm
        if owes_judgment(row, judge):
            verdict, jmeta = ollama(judge, judge_prompt(q, row["reply"]), max_tokens=8)
            if key.startswith("forget|"):
                row["judge"], row["revealed"] = verdict, bool(judge_verdict(verdict))
            else:
                row["judge"], row["judge_meta"], row["correct"] = verdict, jmeta, judge_verdict(verdict)
            progress.put(key, row)
            print(f"  judge {q.qid} {row['arm']} {jmeta['seconds']}s", file=sys.stderr, flush=True)
    out = {"sample_id": conv.sample_id, "rows": [row for _, _, row in main]}
    if forgetting:
        out["forgetting"] = [row for _, _, row in forgetting]
    return out


# --- integrity --------------------------------------------------------------------------------------------------------

def freeze_committed() -> bool:
    if not FREEZE.exists():
        return False
    tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(FREEZE)], capture_output=True)
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(FREEZE)], capture_output=True)
    return tracked.returncode == 0 and clean.returncode == 0


def chosen_split(conversations, name: str) -> List[Conversation]:
    dev, confirm = split(conversations)
    if name == "dev":
        return dev
    if not freeze_committed():
        raise SystemExit(f"CONFIRM stays closed until {FREEZE} is committed (protocol: freeze before CONFIRM).")
    return confirm


# --- analysis (written and frozen before CONFIRM) ---------------------------------------------------------------------

def paired(values: Dict[str, List[Tuple[float, float]]]) -> dict:
    """Pooled difference X − A in points, with a 95% bootstrap interval over conversations (values: conv -> pairs)."""
    convs = sorted(c for c in values if values[c])
    pairs = [p for c in convs for p in values[c]]
    if not pairs:
        return {"n": 0}
    diff = lambda chosen: 100 * sum(x - a for c in chosen for x, a in values[c]) / sum(len(values[c]) for c in chosen)
    rng = random.Random(SEED)
    draws = sorted(diff([rng.choice(convs) for _ in convs]) for _ in range(BOOTSTRAP))
    return {"n": len(pairs), "x": round(100 * sum(x for x, _ in pairs) / len(pairs), 1),
            "a": round(100 * sum(a for _, a in pairs) / len(pairs), 1), "diff": round(diff(convs), 2),
            "low": round(draws[int(0.025 * BOOTSTRAP)], 2), "high": round(draws[int(0.975 * BOOTSTRAP) - 1], 2)}


def coverage_pairs(stage1: dict, x: str, a: str, keep, budget: int, replica: Optional[str] = None):
    """Per conversation, (X, A) full-coverage pairs for questions passing keep(row); X averaged over replicas."""
    out: Dict[str, List[Tuple[float, float]]] = {}
    for conv in stage1["results"]:
        reps = conv["replicas"]
        base = {row["qid"]: row for row in reps["1"]["rows"]}
        pick = [replica] if replica else sorted(reps)
        for qid, row in base.items():
            if not keep(row):
                continue
            value = lambda arm: (float(base[qid]["arms"][arm][str(budget)]["full"]) if arm in ("A0", "A1") else
                                 sum(float({r["qid"]: r for r in reps[k]["rows"]}[qid]["arms"][arm][str(budget)]["full"])
                                     for k in pick) / len(pick))
            out.setdefault(conv["sample_id"], []).append((value(x), value(a)))
    return out


def accuracy_pairs(stage2: dict, x: str, a: str, keep) -> Dict[str, List[Tuple[float, float]]]:
    out: Dict[str, List[Tuple[float, float]]] = {}
    for conv in stage2["results"]:
        by = {}
        for row in conv["rows"]:
            by.setdefault(row["qid"], {})[row["arm"]] = row
        for qid, arms in by.items():
            if x in arms and a in arms and keep(arms[a]):
                out.setdefault(conv["sample_id"], []).append((float(bool(arms[x].get("correct"))),
                                                              float(bool(arms[a].get("correct")))))
    return out


def search_cost(stage1: dict, budget: int) -> dict:
    """Equal tokens are not equal work: per arm, the median packet time, groups visited and member similarities
    read at this budget. Reported beside the gates, never inside them."""
    import statistics

    seen: Dict[str, Dict[str, List[float]]] = {}
    for conv in stage1["results"]:
        for rep in conv["replicas"].values():
            for row in rep["rows"]:
                for arm, by_budget in row["arms"].items():
                    cell = by_budget[str(budget)]
                    if "ms" in cell:
                        for key in ("ms", "visited", "scored"):
                            seen.setdefault(arm, {}).setdefault(key, []).append(cell[key])
    return {arm: dict({f"median_{k}": round(statistics.median(v), 3) for k, v in values.items()},
                      packets=len(values["ms"]))
            for arm, values in sorted(seen.items())}


def analyze(stage1: dict, stage2: dict, conversations: Sequence[Conversation], audit: Optional[dict]) -> dict:
    budget = DECISIVE_BUDGET
    post = {}
    for conv in conversations:
        by_dia = {t.dia_id: t for t in conv.turns}
        for q in conv.questions:
            post[q.qid] = any(e in by_dia and by_dia[e].session > conv.cutoff for e in q.evidence)
    rel = lambda r: r["type"] in RELATIONAL
    rel_post = lambda r: r["type"] in RELATIONAL and post.get(r["qid"], False)
    is_type = lambda t: (lambda r: r["type"] == t)
    pre_type = lambda t: (lambda r: r["type"] == t and not post.get(r["qid"], False))
    local = lambda r: r["type"] in LOCAL
    adv = lambda r: r["type"] == "adversarial"
    cov = lambda x, a, keep, rep=None: paired(coverage_pairs(stage1, x, a, keep, budget, rep))
    acc = lambda x, a, keep: paired(accuracy_pairs(stage2, x, a, keep))
    ok = lambda m, at_least, ci=False: m.get("n", 0) > 0 and m["diff"] >= at_least and (not ci or m["low"] > 0)

    h1 = {"coverage_relational": cov("A1", "A0", rel),
          **{f"coverage_{t}": cov("A1", "A0", is_type(t)) for t in LOCAL},
          "accuracy_relational": acc("A1", "A0", rel), "accuracy_local": acc("A1", "A0", local),
          "accuracy_adversarial": acc("A1", "A0", adv)}
    h1["passed"] = (ok(h1["coverage_relational"], 5, True) and all(ok(h1[f"coverage_{t}"], -2) for t in LOCAL)
                    and ok(h1["accuracy_relational"], 5, True) and ok(h1["accuracy_local"], -3)
                    and ok(h1["accuracy_adversarial"], -3))
    reps = sorted(stage1["results"][0]["replicas"])
    h2 = {"coverage_relational": cov("A2", "A0", rel), "coverage_relational_vs_A1": cov("A2", "A1", rel),
          "coverage_relational_vs_shuffled": cov("A2", "A2-shuffled", rel),
          "coverage_relational_by_replica": {r: cov("A2", "A0", rel, r) for r in reps},
          **{f"coverage_{t}": cov("A2", "A0", is_type(t)) for t in LOCAL},
          "accuracy_relational": acc("A2", "A0", rel), "accuracy_relational_vs_A1": acc("A2", "A1", rel),
          "accuracy_local": acc("A2", "A0", local), "accuracy_adversarial": acc("A2", "A0", adv)}
    h2["passed"] = (ok(h2["coverage_relational"], 5, True) and ok(h2["coverage_relational_vs_A1"], 3)
                    and ok(h2["coverage_relational_vs_shuffled"], 5)
                    and all(ok(m, 0) for m in h2["coverage_relational_by_replica"].values())
                    and all(ok(h2[f"coverage_{t}"], -2) for t in LOCAL)
                    and ok(h2["accuracy_relational"], 5, True) and ok(h2["accuracy_relational_vs_A1"], 3)
                    and ok(h2["accuracy_local"], -3) and ok(h2["accuracy_adversarial"], -3))
    types = RELATIONAL + LOCAL
    h3 = {"coverage_post_relational": cov("A3", "A2", rel_post),
          **{f"coverage_pre_{t}": cov("A3", "A2", pre_type(t)) for t in types},
          "accuracy_post_relational": acc("A3", "A2", rel_post), "accuracy_local": acc("A3", "A2", local),
          "accuracy_adversarial_vs_A0": acc("A3", "A0", adv)}
    h3["passed"] = (ok(h3["coverage_post_relational"], 3, True)
                    and all(ok(h3[f"coverage_pre_{t}"], -2) for t in types)
                    and ok(h3["accuracy_post_relational"], 3) and ok(h3["accuracy_local"], -3)
                    and ok(h3["accuracy_adversarial_vs_A0"], -3))
    revealed = Counter()
    for conv in stage2["results"]:
        for row in conv.get("forgetting", []):
            revealed[row["arm"]] += int(row["revealed"])
    forgetting = {"mechanical": all(c["forgetting"]["passed"] for c in stage1["results"]),
                  "history_outside_gate": all(c["forgetting"].get("history_passed") for c in stage1["results"]),
                  "revealed": dict(revealed),
                  "no_arm_reveals_more_than_A0": all(revealed[a] <= revealed["A0"] for a in ARMS_STAGE2)}
    hard = {"forgetting": forgetting["mechanical"] and forgetting["no_arm_reveals_more_than_A0"],
            "scope": all(s["passed"] for s in stage1["scope"]),
            "rollback": all(r["rollback"] for c in stage1["results"] for r in c["replicas"].values()),
            "judge_audit": audit is not None and audit["agree"] >= 27 and audit["total"] == 30}
    passed_hard = all(hard.values())
    unparsed = sum(1 for c in stage2["results"] for r in c["rows"] if r["type"] != "adversarial"
                   and r.get("correct") is None)
    verdict = {"H1": passed_hard and h1["passed"], "H2_static": passed_hard and h2["passed"],
               "H2_adaptive": passed_hard and h3["passed"]}
    verdict["strong_criterion"] = verdict["H2_static"] and verdict["H2_adaptive"]
    return {"budget": budget, "H1": h1, "H2_static": h2, "H2_adaptive": h3, "forgetting": forgetting,
            "search_cost_outside_gate": search_cost(stage1, budget), "hard_gates": hard, "judge_unparsed_counted_wrong": unparsed, "audit": audit, "verdict": verdict,
            "stage2_status": "INCONCLUSIVE (judge audit)" if not hard["judge_audit"] else "valid"}


def audit_sheet(stage2: dict, conversations: Sequence[Conversation]) -> List[dict]:
    """30 stage-2 judgments drawn by SHA-256, shown without arm or verdict for the maintainer to label."""
    questions = {q.qid: q for c in conversations for q in c.questions}
    rows = [r for c in stage2["results"] for r in c["rows"] if r["type"] != "adversarial" and "judge" in r]
    rows.sort(key=lambda r: sha(f"{NAMESPACE}/audit/{r['qid']}/{r['arm']}"))
    return [{"item": k + 1, "key": sha(f"{NAMESPACE}/audit/{r['qid']}/{r['arm']}")[:12],
             "question": questions[r["qid"]].question, "reference": questions[r["qid"]].answer,
             "response": r["reply"], "your_verdict": ""} for k, r in enumerate(rows[:30])]


def audit_agreement(stage2: dict, labelled: Sequence[dict]) -> dict:
    by_key = {sha(f"{NAMESPACE}/audit/{r['qid']}/{r['arm']}")[:12]: r for c in stage2["results"] for r in c["rows"]}
    agree = sum(1 for item in labelled if item["your_verdict"].strip().upper() in ("CORRECT", "WRONG")
                and (item["your_verdict"].strip().upper() == "CORRECT") == bool(by_key[item["key"]].get("correct")))
    total = sum(1 for item in labelled if item["your_verdict"].strip().upper() in ("CORRECT", "WRONG"))
    return {"agree": agree, "total": total}


HARNESS_NOTES = [
    "Shared rare words: a word in one turn cannot be shared, so the vocabulary whose rarest 5% counts is the words "
    "found in two or more turns.",
    "Discovery drops a resolution whose partition is identical to one already kept: it would only add an identical "
    "step to the walk.",
    "Filling a packet, a turn that does not fit the remaining budget is skipped and the next one is tried.",
    "Adversarial replies are parsed by trimming whitespace and a final period and ignoring case before comparing "
    "with 'Not in memory'.",
    "Evidence ids are split on ';', ',' and spaces; a question naming an id that is not a turn is left out of stage 1.",
    "The write path refuses turns that are only a question; they are missing for every arm (DEV: 2 of 419 and 20 of "
    "689 turns; one DEV open-domain question lost its evidence that way).",
    "When forgetting, a turn whose stored record was shared with a forgotten turn counts as forgotten too.",
    "Observed on DEV, not changed: A2's attach threshold is the 25th percentile of member-to-own-centroid "
    "similarity, and members helped form their centroid, so most later turns stay loose in the frozen arm (conv-26: "
    "145 of 164).",
    "Observed on DEV, not changed: the judge is lenient with partial answers; the human audit (gate) checks it.",
    "The first full DEV stage 2 (started 06:09 on 2026-11-20) hung after the laptop suspended on battery at 07:28 and "
    "was stopped with no output. Stage 2 now saves each answered row and resumes; a hung call is retried. Calls are "
    "deterministic, so neither changes what is measured.",
    "Stage 2 asks the reader for every row of a conversation before the judge grades any. Alternating them made "
    "Ollama reload the 27B reader after each judge call (DEV, 2026-11-20: about 37 s per reader call instead of "
    "10 s). Calls are deterministic and the judge never sees the arm, so the order changes nothing measured. DEV "
    "stage 2 rows answered before this change were judged one at a time.",
    "Added after DEV, outside the gates as written (the four forgetting checks are unchanged): forgetting also "
    "rewrites A3's saved versions and removes forgotten turn ids from its revision log, so a rollback cannot bring "
    "a forgotten turn back; stage 1 reports what is left there (history_passed).",
    "Added after DEV, outside the gates: each packet records its time, groups visited and member similarities "
    "read, since equal tokens are not equal work; stage 1 also times each arm's build. The protocol's full packet "
    "trace (version, nodes visited with scores, each turn's route) is written for every question and arm at the "
    "decisive budget to <out>.traces.jsonl, whose hash goes into the stage 1 output. DEV stage 1 was rerun with "
    "these additions on 2026-11-20: all 16,212 coverage cells, the summary and the scope checks are identical to "
    "the first run, which is kept in artifacts/m1/superseded/. Without the version rewrite, every saved A3 version "
    "on DEV (8 of 8 and 13 of 13) still listed forgotten turns.",
    "Scope isolation checks the project filter (_is_about) inside one owner's store. It is not an authorization "
    "boundary or a multi-user security test; the remote service separates owners into distinct encrypted stores.",
    "No LoCoMo turn reaches the 2,000-character write limit (longest: 454 characters across all 10 conversations, "
    "a length count only). Turns are lost only to the filter for turns that are only a question (see above); "
    "stage 1 reports every refused turn with its reason and the questions whose evidence is missing from the "
    "store, which bound every arm's coverage alike.",
]


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(args) -> None:
    """Write docs/INFINITO_M1_FREEZE.json: everything CONFIRM must run with, fixed after DEV and before CONFIRM."""
    model = Path(args.model)
    dev_outputs = {p.name: file_sha(p) for p in sorted(Path("artifacts/m1").glob("dev_*.json"))}
    a0 = json.loads(Path("artifacts/m1/dev_stage1.json").read_text(encoding="utf-8"))
    a0_rel = a0["summary"][str(DECISIVE_BUDGET)]["A0"]["relational"]["coverage"]
    record = {
        "protocol": "docs/INFINITO_M1_FRACTAL_MEMORY_PROTOCOL_2026-11-20.md",
        "frozen_at": datetime.now().isoformat(timespec="seconds"),
        "harness": {"observatory/m1_fractal_memory.py": file_sha(Path(__file__)),
                    "tests/test_m1_fractal_memory.py": file_sha(Path("tests/test_m1_fractal_memory.py"))},
        "data": {"file": "locomo10.json", "sha256": file_sha(Path(args.data)),
                 "source": "https://github.com/snap-research/locomo", "source_commit": args.data_commit,
                 "licence": "CC BY-NC 4.0"},
        "split": {"dev": [c.sample_id for c in split(load(args.data))[0]],
                  "confirm": [c.sample_id for c in split(load(args.data))[1]]},
        "types": {str(k): v for k, v in TYPES.items()},
        "decisive_budget": {"tokens": DECISIVE_BUDGET if a0_rel < 100 * CEILING_RULE else 512,
                            "rule": f"A0 relational coverage on DEV at 1,024 = {a0_rel}%, below 90%"},
        "stage2_per_type": STAGE2_PER_TYPE,
        "models": {"reader": {"name": args.reader, "digest": model_digest(args.reader)},
                   "judge": {"name": args.judge, "digest": model_digest(args.judge)},
                   "embeddings": {name: file_sha(model / name) for name in ("model.safetensors", "pieces.txt",
                                                                            "scores.f32")}},
        "prompts": {"reader": READER_PROMPT, "judge": JUDGE_PROMPT},
        "dev_outputs_sha256": dev_outputs,
        "harness_notes": HARNESS_NOTES,
    }
    FREEZE.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {FREEZE}; commit it before CONFIRM.", file=sys.stderr)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("stage1", "stage2", "freeze", "audit-sheet", "analyze"))
    parser.add_argument("--data-commit", help="freeze: the LoCoMo repository commit the data came from")
    parser.add_argument("--stage1-file", help="analyze: stage 1 results")
    parser.add_argument("--stage2-file", help="analyze, audit-sheet: stage 2 results")
    parser.add_argument("--audit", help="analyze: the labelled audit sheet (JSON)")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True, help="embeddings folder (potion-multilingual-128M int8)")
    parser.add_argument("--split", choices=("dev", "confirm"), default="dev")
    parser.add_argument("--replicas", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--reader", default="qwen3.8-motor")
    parser.add_argument("--judge")
    parser.add_argument("--limit", type=int, help="stage 2: questions per conversation (DEV timing only)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    if args.stage == "freeze":
        if not (args.judge and args.data_commit):
            raise SystemExit("freeze needs --judge and --data-commit.")
        freeze(args)
        return
    if not args.out:
        raise SystemExit("--out is required.")
    if args.stage in ("audit-sheet", "analyze"):
        stage2 = json.loads(Path(args.stage2_file).read_text(encoding="utf-8"))
        conversations = [c for c in load(args.data) if c.sample_id in stage2["conversations"]]
        if args.stage == "audit-sheet":
            output = audit_sheet(stage2, conversations)
        else:
            stage1 = json.loads(Path(args.stage1_file).read_text(encoding="utf-8"))
            labelled = json.loads(Path(args.audit).read_text(encoding="utf-8")) if args.audit else None
            output = analyze(stage1, stage2, conversations,
                             audit_agreement(stage2, labelled) if labelled else None)
        Path(args.out).write_text(json.dumps(output, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
        return

    from src.infinito3.semantic import ranker_from

    ranker = ranker_from(args.model)
    if ranker is None:
        raise SystemExit(f"No embeddings model in {args.model}.")
    data_hash = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    conversations = chosen_split(load(args.data), args.split)
    result = {"protocol": "docs/INFINITO_M1_FRACTAL_MEMORY_PROTOCOL_2026-11-20.md", "split": args.split,
              "data_sha256": data_hash, "conversations": [c.sample_id for c in conversations],
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    with tempfile.TemporaryDirectory() as folder:
        if args.stage == "stage1":
            convs = []
            traces_path = Path(args.out).with_suffix(".traces.jsonl")
            traces_path.parent.mkdir(parents=True, exist_ok=True)
            with traces_path.open("w", encoding="utf-8") as traces:
                for conv in conversations:
                    print(f"{conv.sample_id}: {len(conv.turns)} turns", file=sys.stderr, flush=True)
                    convs.append(stage1_conversation(conv, ranker, Path(folder), args.replicas, traces))
            result["traces"] = {"file": traces_path.name, "sha256": file_sha(traces_path)}
            result["results"] = convs
            result["summary"] = {str(b): summarize(convs, b) for b in BUDGETS}
            pairs = list(zip(conversations[0::2], conversations[1::2]))  # paired in h order
            (Path(folder) / "scope").mkdir(exist_ok=True)
            result["scope"] = [scope_check(a, b, Path(folder) / "scope") for a, b in pairs]
        else:
            result["reader"] = {"model": args.reader, "digest": model_digest(args.reader)}
            result["judge"] = {"model": args.judge, "digest": model_digest(args.judge) if args.judge else None}
            stay_awake()
            progress = Progress(Path(args.out).with_suffix(".progress.jsonl"))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            result["results"] = [stage2_conversation(c, ranker, Path(folder), args.reader, args.judge, args.limit,
                                                     progress) for c in conversations]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Wrote {args.out}", file=sys.stderr)


# The build and revision functions must never see a question: the tests check these signatures.
BUILDERS = (discover, assemble, signal_graph, time_tree, attach, revise, stream, shuffled, forget_in)


if __name__ == "__main__":
    main()
