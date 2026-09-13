import json
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from .cognitive_loop import CognitiveLoop
from .types import ABComparison, CognitiveRunResult, ContextSource


_TEXT_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _TEXT_RE.sub(" ", str(text).strip().lower())


def _contains(text: str, needle: str) -> bool:
    return _normalize(needle) in _normalize(text)


def _safe_mean(values: Iterable[Optional[float]]) -> Optional[float]:
    concrete = [float(v) for v in values if v is not None]
    return mean(concrete) if concrete else None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class EvaluationExpectation:
    """Deterministic assertions for one evaluation probe.

    Matching is normalized substring matching by design. These metrics are cheap,
    reproducible and transparent; they are not claimed to replace semantic
    evaluation or human judgment.
    """

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
    def judge(
        self,
        scenario: EvaluationScenario,
        comparison: ABComparison,
    ) -> PairwiseJudgeResult:
        ...


class CallablePairwiseJudge:
    """Adapter for plugging in a human, heuristic or LLM judge callback later."""

    def __init__(
        self,
        callback: Callable[[EvaluationScenario, ABComparison], PairwiseJudgeResult],
    ):
        self.callback = callback

    def judge(
        self,
        scenario: EvaluationScenario,
        comparison: ABComparison,
    ) -> PairwiseJudgeResult:
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
        lines = [
            "# INFINITO 3.0 Evaluation Report",
            "",
            f"- Scenarios: {self.summary.scenario_count}",
            f"- Comparable answer probes: {self.summary.comparable_answer_count}",
            f"- Cognitive wins / ties / baseline wins: "
            f"{self.summary.cognitive_wins} / {self.summary.ties} / {self.summary.baseline_wins}",
            f"- Mean baseline answer score: {_fmt(self.summary.mean_baseline_answer_score)}",
            f"- Mean cognitive answer score: {_fmt(self.summary.mean_cognitive_answer_score)}",
            f"- Mean answer lift: {_fmt(self.summary.mean_answer_lift, signed=True)}",
            f"- Mean context score: {_fmt(self.summary.mean_context_score)}",
            f"- Mean context tokens: {_fmt(self.summary.mean_context_tokens)}",
            "",
            "| Scenario | Baseline | Cognitive | Lift | Context |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for result in self.results:
            lines.append(
                f"| {result.scenario.name} | "
                f"{_fmt(result.baseline_metrics.answer_score)} | "
                f"{_fmt(result.cognitive_metrics.answer_score)} | "
                f"{_fmt(result.answer_lift, signed=True)} | "
                f"{_fmt(result.cognitive_metrics.context_score)} |"
            )

        if self.summary.by_tag:
            lines.extend(
                [
                    "",
                    "## By tag",
                    "",
                    "| Tag | Scenarios | Baseline | Cognitive | Lift |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for tag, metrics in sorted(self.summary.by_tag.items()):
                lines.append(
                    f"| {tag} | {int(metrics['scenario_count'] or 0)} | "
                    f"{_fmt(metrics['baseline_answer_score'])} | "
                    f"{_fmt(metrics['cognitive_answer_score'])} | "
                    f"{_fmt(metrics['answer_lift'], signed=True)} |"
                )
        return "\n".join(lines)


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


class DeterministicEvaluator:
    """Transparent evaluator based on fixed expected/forbidden evidence."""

    def score(
        self,
        run: CognitiveRunResult,
        expectation: EvaluationExpectation,
    ) -> RunMetrics:
        answer = run.response.text if run.response is not None else ""
        packet = (
            run.cognitive_decision.context_packet
            if run.cognitive_decision is not None
            else None
        )
        context_text = packet.rendered if packet is not None else ""

        answer_required = self._recall(answer, expectation.answer_contains)
        answer_forbidden = self._rate(answer, expectation.answer_excludes)
        answer_components: List[float] = []
        if answer_required is not None:
            answer_components.append(answer_required)
        if answer_forbidden is not None:
            answer_components.append(1.0 - answer_forbidden)
        answer_score = mean(answer_components) if answer_components else None

        context_required = self._recall(context_text, expectation.context_contains)
        context_forbidden = self._rate(context_text, expectation.context_excludes)
        source_coverage = self._source_coverage(run, expectation.required_sources)
        context_components: List[float] = []
        if context_required is not None:
            context_components.append(context_required)
        if context_forbidden is not None:
            context_components.append(1.0 - context_forbidden)
        if source_coverage is not None:
            context_components.append(source_coverage)
        context_score = mean(context_components) if context_components else None

        usage = run.response.usage if run.response is not None else {}
        return RunMetrics(
            answer_required_recall=answer_required,
            answer_forbidden_rate=answer_forbidden,
            answer_score=answer_score,
            context_required_recall=context_required,
            context_forbidden_rate=context_forbidden,
            source_coverage=source_coverage,
            context_score=context_score,
            context_tokens=packet.estimated_tokens if packet is not None else 0,
            context_items=len(packet.items) if packet is not None else 0,
            duration_ms=run.duration_ms,
            input_tokens=_int_or_none(usage.get("input_tokens")),
            output_tokens=_int_or_none(usage.get("output_tokens")),
            total_tokens=_int_or_none(usage.get("total_tokens")),
            blocked=run.blocked,
        )

    @staticmethod
    def _recall(text: str, required: Sequence[str]) -> Optional[float]:
        if not required:
            return None
        hits = sum(1 for phrase in required if _contains(text, phrase))
        return hits / len(required)

    @staticmethod
    def _rate(text: str, forbidden: Sequence[str]) -> Optional[float]:
        if not forbidden:
            return None
        hits = sum(1 for phrase in forbidden if _contains(text, phrase))
        return hits / len(forbidden)

    @staticmethod
    def _source_coverage(
        run: CognitiveRunResult,
        required_sources: Sequence[ContextSource],
    ) -> Optional[float]:
        if not required_sources:
            return None
        packet = (
            run.cognitive_decision.context_packet
            if run.cognitive_decision is not None
            else None
        )
        present = {item.source for item in packet.items} if packet is not None else set()
        hits = sum(1 for source in required_sources if source in present)
        return hits / len(required_sources)


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class EvaluationHarness:
    """Run isolated, reproducible A/B probes over CognitiveLoop.

    A fresh loop is created for every scenario. Setup turns are executed through
    cognition so INFINITO can build memory/goals. The final probe uses
    CognitiveLoop.compare(), giving baseline and cognitive variants the exact
    same visible pre-probe conversation state. This measures the value of hidden
    cognitive state at the probe without cross-scenario contamination.
    """

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

    def run(
        self,
        scenarios: Sequence[EvaluationScenario],
    ) -> EvaluationReport:
        results = [self.run_scenario(scenario) for scenario in scenarios]
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

        setup_failures: List[str] = []
        for index, turn in enumerate(scenario.setup_turns):
            setup = loop.turn(
                turn,
                use_cognition=True,
                context_budget_tokens=scenario.context_budget_tokens,
                top_k=scenario.top_k,
                max_output_tokens=scenario.max_output_tokens,
            )
            if setup.blocked:
                setup_failures.append(
                    f"setup[{index}] blocked: {setup.block_reason or 'unknown'}"
                )
            elif setup.response is None:
                setup_failures.append(f"setup[{index}] produced no response")

        comparison = loop.compare(
            scenario.probe,
            context_budget_tokens=scenario.context_budget_tokens,
            top_k=scenario.top_k,
            max_output_tokens=scenario.max_output_tokens,
            commit_cognitive_response=False,
        )
        baseline_metrics = self.evaluator.score(
            comparison.baseline,
            scenario.expectation,
        )
        cognitive_metrics = self.evaluator.score(
            comparison.cognitive,
            scenario.expectation,
        )

        answer_lift: Optional[float] = None
        if (
            baseline_metrics.answer_score is not None
            and cognitive_metrics.answer_score is not None
        ):
            answer_lift = (
                cognitive_metrics.answer_score - baseline_metrics.answer_score
            )

        judge_result = (
            self.judge.judge(scenario, comparison) if self.judge is not None else None
        )

        return ScenarioResult(
            scenario=scenario,
            comparison=comparison,
            baseline_metrics=baseline_metrics,
            cognitive_metrics=cognitive_metrics,
            answer_lift=answer_lift,
            judge=judge_result,
            setup_failures=setup_failures,
        )

    def _summarize(self, results: Sequence[ScenarioResult]) -> EvaluationSummary:
        comparable = [r for r in results if r.answer_lift is not None]
        wins = sum(1 for r in comparable if r.answer_lift > self.win_epsilon)
        losses = sum(1 for r in comparable if r.answer_lift < -self.win_epsilon)
        ties = len(comparable) - wins - losses

        token_deltas: List[float] = []
        for result in results:
            baseline_total = result.baseline_metrics.total_tokens
            cognitive_total = result.cognitive_metrics.total_tokens
            if baseline_total is not None and cognitive_total is not None:
                token_deltas.append(float(cognitive_total - baseline_total))

        by_tag: Dict[str, Dict[str, Optional[float]]] = {}
        tags = sorted({tag for result in results for tag in result.scenario.tags})
        for tag in tags:
            tagged = [result for result in results if tag in result.scenario.tags]
            by_tag[tag] = {
                "scenario_count": float(len(tagged)),
                "baseline_answer_score": _safe_mean(
                    r.baseline_metrics.answer_score for r in tagged
                ),
                "cognitive_answer_score": _safe_mean(
                    r.cognitive_metrics.answer_score for r in tagged
                ),
                "answer_lift": _safe_mean(r.answer_lift for r in tagged),
            }

        return EvaluationSummary(
            scenario_count=len(results),
            comparable_answer_count=len(comparable),
            cognitive_wins=wins,
            ties=ties,
            baseline_wins=losses,
            mean_baseline_answer_score=_safe_mean(
                r.baseline_metrics.answer_score for r in results
            ),
            mean_cognitive_answer_score=_safe_mean(
                r.cognitive_metrics.answer_score for r in results
            ),
            mean_answer_lift=_safe_mean(r.answer_lift for r in results),
            mean_context_score=_safe_mean(
                r.cognitive_metrics.context_score for r in results
            ),
            mean_context_tokens=_safe_mean(
                float(r.cognitive_metrics.context_tokens) for r in results
            ),
            mean_latency_delta_ms=_safe_mean(
                r.cognitive_metrics.duration_ms - r.baseline_metrics.duration_ms
                for r in results
            ),
            mean_total_token_delta=mean(token_deltas) if token_deltas else None,
            by_tag=by_tag,
        )


def standard_evaluation_suite() -> Tuple[EvaluationScenario, ...]:
    """Small fixed suite covering the first cognitive failure modes.

    These scenarios are intentionally generic. They can be run with the
    deterministic test adapter or a real LLM. The exact-match metrics stay
    transparent; optional semantic/human judging can be layered on separately.
    """

    return (
        EvaluationScenario(
            name="long_term_identity",
            description="Recall identity after it falls out of short-term history.",
            tags=("factual_continuity", "long_term_memory"),
            history_limit=2,
            setup_turns=(
                "Me llamo Alicia.",
                "Vale.",
                "Perfecto.",
            ),
            probe="¿Cómo me llamo?",
            expectation=EvaluationExpectation(
                answer_contains=("Alicia",),
                context_contains=("Alicia",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
        EvaluationScenario(
            name="contradiction_resolution",
            description="Use the newest exclusive fact and suppress the superseded one.",
            tags=("contradictions", "factual_continuity"),
            history_limit=0,
            setup_turns=(
                "Vivo en Sevilla.",
                "Vivo en Bilbao.",
            ),
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
            description="Recover an active goal with no visible conversation history.",
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
            description="Retrieve task-relevant user state without unrelated preference noise.",
            tags=("relevance", "user_model"),
            history_limit=0,
            setup_turns=(
                "Me gusta el café.",
                "Me gusta mi bici Trek Session.",
            ),
            probe="¿Qué bici uso?",
            expectation=EvaluationExpectation(
                answer_contains=("Trek Session",),
                context_contains=("Trek Session",),
                context_excludes=("café",),
                required_sources=(ContextSource.USER_MODEL,),
            ),
        ),
    )
