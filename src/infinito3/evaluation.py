import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime
from enum import Enum
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .cognitive_loop import CognitiveLoop
from .types import ABComparison, CognitiveRunResult, ContextSource


_WS_RE = re.compile(r"\s+")


def _norm(value: Any) -> str:
    return _WS_RE.sub(" ", str(value).strip().lower())


def _contains(text: str, phrase: str) -> bool:
    return _norm(phrase) in _norm(text)


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    concrete = [float(v) for v in values if v is not None]
    return mean(concrete) if concrete else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class EvaluationExpectation:
    answer_contains: Tuple[str, ...] = ()
    answer_excludes: Tuple[str, ...] = ()
    context_contains: Tuple[str, ...] = ()
    context_excludes: Tuple[str, ...] = ()
    required_sources: Tuple[ContextSource, ...] = ()


@dataclass(frozen=True)
class EvaluationScenario:
    name: str
    probe: str
    setup_turns: Tuple[str, ...] = ()
    expectation: EvaluationExpectation = field(default_factory=EvaluationExpectation)
    description: str = ""
    tags: Tuple[str, ...] = ()
    history_limit: Optional[int] = None
    context_budget_tokens: int = 1200
    top_k: int = 8
    max_output_tokens: Optional[int] = None


@dataclass
class RunMetrics:
    answer_required_recall: Optional[float] = None
    answer_forbidden_rate: Optional[float] = None
    answer_score: Optional[float] = None
    context_required_recall: Optional[float] = None
    context_forbidden_rate: Optional[float] = None
    source_coverage: Optional[float] = None
    context_score: Optional[float] = None
    context_tokens: int = 0
    context_items: int = 0
    duration_ms: float = 0.0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    blocked: bool = False


@dataclass
class PairwiseJudgeResult:
    winner: str
    score: float
    rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PairwiseJudge(Protocol):
    def judge(self, scenario: EvaluationScenario, comparison: ABComparison) -> PairwiseJudgeResult:
        ...


class CallablePairwiseJudge:
    def __init__(self, callback: Callable[[EvaluationScenario, ABComparison], PairwiseJudgeResult]):
        self.callback = callback

    def judge(self, scenario: EvaluationScenario, comparison: ABComparison) -> PairwiseJudgeResult:
        return self.callback(scenario, comparison)


@dataclass
class ScenarioResult:
    scenario: EvaluationScenario
    comparison: ABComparison
    baseline_metrics: RunMetrics
    cognitive_metrics: RunMetrics
    answer_lift: Optional[float]
    judge: Optional[PairwiseJudgeResult] = None
    setup_failures: List[str] = field(default_factory=list)


@dataclass
class EvaluationSummary:
    scenario_count: int
    comparable_answer_count: int
    cognitive_wins: int
    ties: int
    baseline_wins: int
    mean_baseline_answer_score: Optional[float]
    mean_cognitive_answer_score: Optional[float]
    mean_answer_lift: Optional[float]
    mean_context_score: Optional[float]
    mean_context_tokens: Optional[float]
    mean_latency_delta_ms: Optional[float]
    mean_total_token_delta: Optional[float]
    by_tag: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    results: List[ScenarioResult]
    summary: EvaluationSummary
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        s = self.summary
        lines = [
            "# INFINITO 3.0 Evaluation Report",
            "",
            f"- Scenarios: {s.scenario_count}",
            f"- Comparable answer probes: {s.comparable_answer_count}",
            f"- Cognitive wins / ties / baseline wins: {s.cognitive_wins} / {s.ties} / {s.baseline_wins}",
            f"- Mean baseline answer score: {_fmt(s.mean_baseline_answer_score)}",
            f"- Mean cognitive answer score: {_fmt(s.mean_cognitive_answer_score)}",
            f"- Mean answer lift: {_fmt(s.mean_answer_lift, signed=True)}",
            f"- Mean context score: {_fmt(s.mean_context_score)}",
            f"- Mean context tokens: {_fmt(s.mean_context_tokens)}",
            "",
            "| Scenario | Baseline | Cognitive | Lift | Context |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for r in self.results:
            lines.append(
                f"| {r.scenario.name} | {_fmt(r.baseline_metrics.answer_score)} | "
                f"{_fmt(r.cognitive_metrics.answer_score)} | {_fmt(r.answer_lift, signed=True)} | "
                f"{_fmt(r.cognitive_metrics.context_score)} |"
            )
        if s.by_tag:
            lines += ["", "## By tag", "", "| Tag | Scenarios | Baseline | Cognitive | Lift |",
                      "| --- | ---: | ---: | ---: | ---: |"]
            for tag, m in sorted(s.by_tag.items()):
                lines.append(
                    f"| {tag} | {int(m['scenario_count'] or 0)} | {_fmt(m['baseline_answer_score'])} | "
                    f"{_fmt(m['cognitive_answer_score'])} | {_fmt(m['answer_lift'], signed=True)} |"
                )
        return "\n".join(lines)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


class DeterministicEvaluator:
    """Cheap transparent scorer; semantic/human judging is a separate plug-in."""

    def score(self, run: CognitiveRunResult, expectation: EvaluationExpectation) -> RunMetrics:
        answer = run.response.text if run.response is not None else ""
        packet = run.cognitive_decision.context_packet if run.cognitive_decision is not None else None
        context = packet.rendered if packet is not None else ""

        ar = self._recall(answer, expectation.answer_contains)
        af = self._rate(answer, expectation.answer_excludes)
        cr = self._recall(context, expectation.context_contains)
        cf = self._rate(context, expectation.context_excludes)
        sc = self._source_coverage(run, expectation.required_sources)

        answer_parts = ([ar] if ar is not None else []) + ([1.0 - af] if af is not None else [])
        context_parts = (
            ([cr] if cr is not None else [])
            + ([1.0 - cf] if cf is not None else [])
            + ([sc] if sc is not None else [])
        )
        usage = run.response.usage if run.response is not None else {}
        return RunMetrics(
            answer_required_recall=ar,
            answer_forbidden_rate=af,
            answer_score=mean(answer_parts) if answer_parts else None,
            context_required_recall=cr,
            context_forbidden_rate=cf,
            source_coverage=sc,
            context_score=mean(context_parts) if context_parts else None,
            context_tokens=packet.estimated_tokens if packet is not None else 0,
            context_items=len(packet.items) if packet is not None else 0,
            duration_ms=run.duration_ms,
            input_tokens=_int(usage.get("input_tokens")),
            output_tokens=_int(usage.get("output_tokens")),
            total_tokens=_int(usage.get("total_tokens")),
            blocked=run.blocked,
        )

    @staticmethod
    def _recall(text: str, phrases: Sequence[str]) -> Optional[float]:
        return None if not phrases else sum(_contains(text, p) for p in phrases) / len(phrases)

    @staticmethod
    def _rate(text: str, phrases: Sequence[str]) -> Optional[float]:
        return None if not phrases else sum(_contains(text, p) for p in phrases) / len(phrases)

    @staticmethod
    def _source_coverage(run: CognitiveRunResult, sources: Sequence[ContextSource]) -> Optional[float]:
        if not sources:
            return None
        packet = run.cognitive_decision.context_packet if run.cognitive_decision is not None else None
        present = {item.source for item in packet.items} if packet is not None else set()
        return sum(source in present for source in sources) / len(sources)


def _int(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class EvaluationHarness:
    """Fresh loop per scenario; final probe is same-history baseline vs cognition."""

    def __init__(
        self,
        loop_factory: Callable[[], CognitiveLoop],
        *,
        evaluator: Optional[DeterministicEvaluator] = None,
        judge: Optional[PairwiseJudge] = None,
        win_epsilon: float = 0.05,
    ):
        if win_epsilon < 0:
            raise ValueError("win_epsilon must be >= 0")
        self.loop_factory = loop_factory
        self.evaluator = evaluator or DeterministicEvaluator()
        self.judge = judge
        self.win_epsilon = win_epsilon

    def run(self, scenarios: Sequence[EvaluationScenario]) -> EvaluationReport:
        results = [self.run_scenario(s) for s in scenarios]
        return EvaluationReport(
            results=results,
            summary=self._summarize(results),
            metadata={
                "isolation": "fresh_loop_per_scenario",
                "comparison": "same_pre_probe_history",
                "deterministic_matching": "normalized_substring",
                "judge_enabled": self.judge is not None,
            },
        )

    def run_scenario(self, scenario: EvaluationScenario) -> ScenarioResult:
        loop = self.loop_factory()
        if scenario.history_limit is not None:
            if scenario.history_limit < 0:
                raise ValueError("scenario history_limit must be >= 0")
            loop.history_limit = scenario.history_limit
            loop.reset_history()

        failures: List[str] = []
        for i, text in enumerate(scenario.setup_turns):
            setup = loop.turn(
                text,
                use_cognition=True,
                context_budget_tokens=scenario.context_budget_tokens,
                top_k=scenario.top_k,
                max_output_tokens=scenario.max_output_tokens,
            )
            if setup.blocked:
                failures.append(f"setup[{i}] blocked: {setup.block_reason or 'unknown'}")
            elif setup.response is None:
                failures.append(f"setup[{i}] produced no response")

        comparison = loop.compare(
            scenario.probe,
            context_budget_tokens=scenario.context_budget_tokens,
            top_k=scenario.top_k,
            max_output_tokens=scenario.max_output_tokens,
            commit_cognitive_response=False,
        )
        baseline = self.evaluator.score(comparison.baseline, scenario.expectation)
        cognitive = self.evaluator.score(comparison.cognitive, scenario.expectation)
        lift = (
            cognitive.answer_score - baseline.answer_score
            if cognitive.answer_score is not None and baseline.answer_score is not None
            else None
        )
        judged = self.judge.judge(scenario, comparison) if self.judge is not None else None
        return ScenarioResult(scenario, comparison, baseline, cognitive, lift, judged, failures)

    def _summarize(self, results: Sequence[ScenarioResult]) -> EvaluationSummary:
        comparable = [r for r in results if r.answer_lift is not None]
        wins = sum(r.answer_lift > self.win_epsilon for r in comparable)
        losses = sum(r.answer_lift < -self.win_epsilon for r in comparable)
        ties = len(comparable) - wins - losses

        token_deltas = [
            float(r.cognitive_metrics.total_tokens - r.baseline_metrics.total_tokens)
            for r in results
            if r.cognitive_metrics.total_tokens is not None and r.baseline_metrics.total_tokens is not None
        ]
        by_tag: Dict[str, Dict[str, Optional[float]]] = {}
        for tag in sorted({tag for r in results for tag in r.scenario.tags}):
            group = [r for r in results if tag in r.scenario.tags]
            by_tag[tag] = {
                "scenario_count": float(len(group)),
                "baseline_answer_score": _avg(r.baseline_metrics.answer_score for r in group),
                "cognitive_answer_score": _avg(r.cognitive_metrics.answer_score for r in group),
                "answer_lift": _avg(r.answer_lift for r in group),
            }

        return EvaluationSummary(
            scenario_count=len(results),
            comparable_answer_count=len(comparable),
            cognitive_wins=wins,
            ties=ties,
            baseline_wins=losses,
            mean_baseline_answer_score=_avg(r.baseline_metrics.answer_score for r in results),
            mean_cognitive_answer_score=_avg(r.cognitive_metrics.answer_score for r in results),
            mean_answer_lift=_avg(r.answer_lift for r in results),
            mean_context_score=_avg(r.cognitive_metrics.context_score for r in results),
            mean_context_tokens=_avg(float(r.cognitive_metrics.context_tokens) for r in results),
            mean_latency_delta_ms=_avg(
                r.cognitive_metrics.duration_ms - r.baseline_metrics.duration_ms for r in results
            ),
            mean_total_token_delta=mean(token_deltas) if token_deltas else None,
            by_tag=by_tag,
        )


def standard_evaluation_suite() -> Tuple[EvaluationScenario, ...]:
    return (
        EvaluationScenario(
            name="long_term_identity",
            description="Recall identity after it falls out of short-term history.",
            tags=("factual_continuity", "long_term_memory"),
            history_limit=2,
            setup_turns=("Me llamo Alicia.", "Vale.", "Perfecto."),
            probe="¿Cómo me llamo?",
            expectation=EvaluationExpectation(
                answer_contains=("Alicia",),
                context_contains=("Alicia",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="contradiction_resolution",
            description="Use newest exclusive fact and suppress superseded value.",
            tags=("contradictions", "factual_continuity"),
            history_limit=0,
            setup_turns=("Vivo en Sevilla.", "Vivo en Bilbao."),
            probe="¿Dónde vivo ahora?",
            expectation=EvaluationExpectation(
                answer_contains=("Bilbao",),
                answer_excludes=("Sevilla",),
                context_contains=("Bilbao",),
                context_excludes=("Sevilla",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="goal_continuity",
            description="Recover active goal with no visible conversation history.",
            tags=("goal_continuity",),
            history_limit=0,
            setup_turns=("Mañana tengo que entrenar descenso a las 10.",),
            probe="¿Qué tengo pendiente?",
            expectation=EvaluationExpectation(
                answer_contains=("entrenar",),
                context_contains=("entrenar descenso",),
                required_sources=(ContextSource.GOAL,),
            ),
        ),
        EvaluationScenario(
            name="relevance_filter",
            description="Retrieve relevant user state without unrelated preference noise.",
            tags=("relevance", "user_model"),
            history_limit=0,
            setup_turns=("Me gusta el café.", "Me gusta mi bici Trek Session."),
            probe="¿Qué bici uso?",
            expectation=EvaluationExpectation(
                answer_contains=("Trek Session",),
                context_contains=("Trek Session",),
                context_excludes=("café",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
    )
