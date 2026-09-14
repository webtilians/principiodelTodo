import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .cognitive_loop import CognitiveLoop
from .evaluation import DeterministicEvaluator, EvaluationExpectation, RunMetrics
from .types import CognitiveRunResult


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


def _avg(values: Iterable[Optional[float]]) -> Optional[float]:
    concrete = [float(v) for v in values if v is not None]
    return mean(concrete) if concrete else None


def _fmt(value: Optional[float], *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.3f}" if signed else f"{value:.3f}"


class MutableClock:
    """Deterministic clock shared by a trajectory's cognitive components."""

    def __init__(self, current: datetime):
        self.current = current

    def __call__(self) -> datetime:
        return self.current

    def advance(self, *, hours: float = 0.0) -> datetime:
        self.current = self.current + timedelta(hours=float(hours))
        return self.current


@dataclass(frozen=True)
class TrajectoryStep:
    """One user turn, optionally preceded by simulated elapsed time.

    An expectation marks the turn as an evaluation checkpoint. Every actual
    user turn is executed independently by the baseline and cognitive loops and
    both responses are committed to their own short-term histories.
    """

    user_text: Optional[str] = None
    label: str = ""
    expectation: Optional[EvaluationExpectation] = None
    advance_hours: float = 0.0
    context_budget_tokens: Optional[int] = None
    top_k: Optional[int] = None
    max_output_tokens: Optional[int] = None
    tags: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TrajectoryScenario:
    name: str
    steps: Tuple[TrajectoryStep, ...]
    description: str = ""
    tags: Tuple[str, ...] = ()
    start_at: datetime = datetime(2026, 9, 14, 9, 0, 0)
    history_limit: int = 6
    context_budget_tokens: int = 700
    top_k: int = 10
    max_output_tokens: Optional[int] = 96


@dataclass
class TrajectoryStepResult:
    index: int
    label: str
    user_text: str
    simulated_at: datetime
    baseline: CognitiveRunResult
    cognitive: CognitiveRunResult
    expectation: Optional[EvaluationExpectation] = None
    baseline_metrics: Optional[RunMetrics] = None
    cognitive_metrics: Optional[RunMetrics] = None
    answer_lift: Optional[float] = None

    @property
    def is_probe(self) -> bool:
        return self.expectation is not None


@dataclass
class TrajectoryScenarioResult:
    scenario: TrajectoryScenario
    steps: List[TrajectoryStepResult]
    final_active_memories: int
    final_open_goals: int
    baseline_total_tokens: int
    cognitive_total_tokens: int
    cumulative_context_tokens: int

    @property
    def probes(self) -> List[TrajectoryStepResult]:
        return [step for step in self.steps if step.is_probe]


@dataclass
class TrajectorySummary:
    trajectory_count: int
    user_turn_count: int
    probe_count: int
    cognitive_wins: int
    ties: int
    baseline_wins: int
    mean_baseline_answer_score: Optional[float]
    mean_cognitive_answer_score: Optional[float]
    mean_answer_lift: Optional[float]
    mean_context_score: Optional[float]
    baseline_total_tokens: int
    cognitive_total_tokens: int
    total_token_delta: int
    cumulative_context_tokens: int
    final_active_memories: int
    final_open_goals: int


@dataclass
class TrajectoryReport:
    results: List[TrajectoryScenarioResult]
    summary: TrajectorySummary
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return _jsonable(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        s = self.summary
        lines = [
            "# INFINITO 3.0 Independent Trajectory Report",
            "",
            f"- Trajectories: {s.trajectory_count}",
            f"- User turns: {s.user_turn_count}",
            f"- Evaluation probes: {s.probe_count}",
            f"- Cognitive wins / ties / baseline wins: {s.cognitive_wins} / {s.ties} / {s.baseline_wins}",
            f"- Mean baseline answer score: {_fmt(s.mean_baseline_answer_score)}",
            f"- Mean cognitive answer score: {_fmt(s.mean_cognitive_answer_score)}",
            f"- Mean answer lift: {_fmt(s.mean_answer_lift, signed=True)}",
            f"- Mean cognitive context score: {_fmt(s.mean_context_score)}",
            f"- Baseline / cognitive provider tokens: {s.baseline_total_tokens} / {s.cognitive_total_tokens}",
            f"- Total provider token delta: {s.total_token_delta:+d}",
            f"- Cumulative INFINITO context tokens: {s.cumulative_context_tokens}",
            f"- Final active memories / open goals: {s.final_active_memories} / {s.final_open_goals}",
            "",
            "| Trajectory | Turn | Probe | Baseline | Cognitive | Lift | Context |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
        for result in self.results:
            for step in result.probes:
                lines.append(
                    f"| {result.scenario.name} | {step.index + 1} | {step.label or step.user_text[:42]} | "
                    f"{_fmt(step.baseline_metrics.answer_score if step.baseline_metrics else None)} | "
                    f"{_fmt(step.cognitive_metrics.answer_score if step.cognitive_metrics else None)} | "
                    f"{_fmt(step.answer_lift, signed=True)} | "
                    f"{_fmt(step.cognitive_metrics.context_score if step.cognitive_metrics else None)} |"
                )
        return "\n".join(lines)


LoopPairFactory = Callable[[MutableClock, TrajectoryScenario], Tuple[CognitiveLoop, CognitiveLoop]]


class TrajectoryEvaluationHarness:
    """Evaluate full independent baseline and cognitive conversation trajectories.

    Unlike EvaluationHarness, this harness does not reset both variants to the
    same pre-probe history. The baseline and cognitive loops start from the same
    empty state, then evolve independently for the entire scenario. Every user
    turn is delivered to both arms, while each arm commits its own assistant
    response. This measures the accumulated effect of cognition in a more
    realistic conversational process.
    """

    def __init__(
        self,
        loop_pair_factory: LoopPairFactory,
        *,
        evaluator: Optional[DeterministicEvaluator] = None,
        win_epsilon: float = 0.05,
    ):
        if win_epsilon < 0:
            raise ValueError("win_epsilon must be >= 0")
        self.loop_pair_factory = loop_pair_factory
        self.evaluator = evaluator or DeterministicEvaluator()
        self.win_epsilon = win_epsilon

    def run(self, scenarios: Sequence[TrajectoryScenario]) -> TrajectoryReport:
        results = [self.run_scenario(scenario) for scenario in scenarios]
        probes = [step for result in results for step in result.probes]
        comparable = [step for step in probes if step.answer_lift is not None]
        wins = sum(step.answer_lift > self.win_epsilon for step in comparable)
        losses = sum(step.answer_lift < -self.win_epsilon for step in comparable)
        ties = len(comparable) - wins - losses
        baseline_tokens = sum(result.baseline_total_tokens for result in results)
        cognitive_tokens = sum(result.cognitive_total_tokens for result in results)

        summary = TrajectorySummary(
            trajectory_count=len(results),
            user_turn_count=sum(len(result.steps) for result in results),
            probe_count=len(probes),
            cognitive_wins=wins,
            ties=ties,
            baseline_wins=losses,
            mean_baseline_answer_score=_avg(
                step.baseline_metrics.answer_score if step.baseline_metrics else None
                for step in probes
            ),
            mean_cognitive_answer_score=_avg(
                step.cognitive_metrics.answer_score if step.cognitive_metrics else None
                for step in probes
            ),
            mean_answer_lift=_avg(step.answer_lift for step in probes),
            mean_context_score=_avg(
                step.cognitive_metrics.context_score if step.cognitive_metrics else None
                for step in probes
            ),
            baseline_total_tokens=baseline_tokens,
            cognitive_total_tokens=cognitive_tokens,
            total_token_delta=cognitive_tokens - baseline_tokens,
            cumulative_context_tokens=sum(result.cumulative_context_tokens for result in results),
            final_active_memories=sum(result.final_active_memories for result in results),
            final_open_goals=sum(result.final_open_goals for result in results),
        )
        return TrajectoryReport(
            results=results,
            summary=summary,
            metadata={
                "comparison": "independent_full_trajectories",
                "same_user_turns": True,
                "assistant_histories": "independent",
                "deterministic_matching": "normalized_substring",
            },
        )

    def run_scenario(self, scenario: TrajectoryScenario) -> TrajectoryScenarioResult:
        if scenario.history_limit < 0:
            raise ValueError("trajectory history_limit must be >= 0")
        clock = MutableClock(scenario.start_at)
        baseline_loop, cognitive_loop = self.loop_pair_factory(clock, scenario)
        for loop in (baseline_loop, cognitive_loop):
            loop.history_limit = scenario.history_limit
            loop.reset_history()

        steps: List[TrajectoryStepResult] = []
        baseline_total_tokens = 0
        cognitive_total_tokens = 0
        cumulative_context_tokens = 0

        for index, step in enumerate(scenario.steps):
            if step.advance_hours:
                clock.advance(hours=step.advance_hours)
            if step.user_text is None:
                continue

            budget = step.context_budget_tokens or scenario.context_budget_tokens
            top_k = step.top_k or scenario.top_k
            max_output = (
                step.max_output_tokens
                if step.max_output_tokens is not None
                else scenario.max_output_tokens
            )
            baseline = baseline_loop.turn(
                step.user_text,
                use_cognition=False,
                context_budget_tokens=budget,
                top_k=top_k,
                max_output_tokens=max_output,
            )
            cognitive = cognitive_loop.turn(
                step.user_text,
                use_cognition=True,
                context_budget_tokens=budget,
                top_k=top_k,
                max_output_tokens=max_output,
            )

            baseline_total_tokens += self._run_total_tokens(baseline)
            cognitive_total_tokens += self._run_total_tokens(cognitive)
            packet = cognitive.cognitive_decision.context_packet if cognitive.cognitive_decision else None
            if packet is not None:
                cumulative_context_tokens += packet.estimated_tokens

            baseline_metrics = None
            cognitive_metrics = None
            lift = None
            if step.expectation is not None:
                baseline_metrics = self.evaluator.score(baseline, step.expectation)
                cognitive_metrics = self.evaluator.score(cognitive, step.expectation)
                if (
                    baseline_metrics.answer_score is not None
                    and cognitive_metrics.answer_score is not None
                ):
                    lift = cognitive_metrics.answer_score - baseline_metrics.answer_score

            steps.append(
                TrajectoryStepResult(
                    index=index,
                    label=step.label,
                    user_text=step.user_text,
                    simulated_at=clock(),
                    baseline=baseline,
                    cognitive=cognitive,
                    expectation=step.expectation,
                    baseline_metrics=baseline_metrics,
                    cognitive_metrics=cognitive_metrics,
                    answer_lift=lift,
                )
            )

        return TrajectoryScenarioResult(
            scenario=scenario,
            steps=steps,
            final_active_memories=self._memory_count(cognitive_loop),
            final_open_goals=self._goal_count(cognitive_loop),
            baseline_total_tokens=baseline_total_tokens,
            cognitive_total_tokens=cognitive_total_tokens,
            cumulative_context_tokens=cumulative_context_tokens,
        )

    @staticmethod
    def _run_total_tokens(run: CognitiveRunResult) -> int:
        usage = run.response.usage if run.response is not None else {}
        try:
            return int(usage.get("total_tokens") or 0)
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _memory_count(loop: CognitiveLoop) -> int:
        getter = getattr(loop.engine.memory_store, "all", None)
        if getter is None:
            return 0
        try:
            return len(getter())
        except Exception:
            return 0

    @staticmethod
    def _goal_count(loop: CognitiveLoop) -> int:
        getter = getattr(loop.engine.goal_engine, "all", None)
        if getter is None:
            return 0
        try:
            return sum(not goal.completed for goal in getter())
        except Exception:
            return 0
