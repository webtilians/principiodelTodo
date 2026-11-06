from datetime import datetime
from typing import List, Optional, Sequence

from .forgetting import ForgetRequestDetector, OverlapForgetPlanner
from .generalized_context_builder import GeneralizedContextBuilder
from .goals import SimpleGoalEngine
from .interfaces import (
    ContextBuilder,
    EmbeddingProvider,
    GoalEngine,
    MemoryGate,
    MemoryStore,
    SafetyFilter,
)
from .memory import InMemoryMemoryStore, RuleBasedMemoryGate, is_question, split_sentences
from .safety import SensitiveInformationFilter
from .types import CognitiveDecision, ContextPacket, ConversationTurn, MemoryRecord, SafetyDecision, SafetyLevel


class CognitiveEngine:
    """Orchestrates the INFINITO 3.0 cognitive layer.

    Processing order is deliberate:
    1. Safety inspection
    2. Forget requests: delete what they name and store nothing of them
    3. Retrieval from existing memory
    4. Memory-gate evaluation, sentence by sentence
    5. Goal extraction
    6. Context construction from pre-write state + active goals
    7. Persistence, only when policy allows it

    No LLM provider or UI dependency belongs here; a forget planner may use
    one, through the adapter it is given.
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        memory_gate: Optional[MemoryGate] = None,
        safety_filter: Optional[SafetyFilter] = None,
        goal_engine: Optional[GoalEngine] = None,
        context_builder: Optional[ContextBuilder] = None,
        forget_detector: Optional[ForgetRequestDetector] = None,
        forget_planner=None,
        forget_candidates: Optional[int] = None,
    ):
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.memory_gate = memory_gate or RuleBasedMemoryGate()
        self.safety_filter = safety_filter or SensitiveInformationFilter()
        self.goal_engine = goal_engine or SimpleGoalEngine()
        self.context_builder = context_builder or GeneralizedContextBuilder(
            memory_store=self.memory_store,
            goal_engine=self.goal_engine,
        )
        self.forget_detector = forget_detector or ForgetRequestDetector()
        self.forget_planner = forget_planner or OverlapForgetPlanner()
        # How many memories a forget request is checked against; None means all.
        self.forget_candidates = forget_candidates

    @classmethod
    def persistent(
        cls,
        db_path: str = "data/infinito3_memory.db",
        embedding_provider: Optional[EmbeddingProvider] = None,
        **kwargs,
    ) -> "CognitiveEngine":
        """Create an engine backed by the real persistent v3 memory."""
        from .persistent_memory import SQLiteCognitiveMemoryStore

        store = SQLiteCognitiveMemoryStore(
            path=db_path,
            embedding_provider=embedding_provider,
        )
        return cls(memory_store=store, **kwargs)

    def process(
        self,
        text: str,
        top_k: int = 5,
        *,
        context_budget_tokens: int = 1200,
        recent_turns: Optional[Sequence[ConversationTurn]] = None,
        build_context: bool = True,
        said_at: Optional[datetime] = None,
        keep_contact_details: bool = False,
        metadata: Optional[dict] = None,
    ) -> CognitiveDecision:
        """Handle one user message. ``said_at`` is when the user said it, if not now.

        ``keep_contact_details`` stores emails and phone numbers as they were said, for a memory that belongs to one
        person (mcp_server.py); ``metadata`` goes with what gets stored.
        """
        safety = self.safety_filter.inspect(text)

        # Secrets are blocked before any cognitive subsystem receives them.
        if safety.level == SafetyLevel.FORBIDDEN:
            return CognitiveDecision(
                input_text=text,
                safety=safety,
                gate=None,
                context=[],
                context_packet=None,
            )

        keep_contact = keep_contact_details and safety.level == SafetyLevel.SENSITIVE
        query = text if keep_contact else (safety.redacted_text if safety.redacted_text else text)

        # A forget request is never stored: it usually repeats what it names.
        if self.forget_detector.is_request(query):
            decision = self._forget(text, query, safety, top_k, context_budget_tokens,
                                    recent_turns, build_context, said_at)
            if decision is not None:
                return decision

        # Retrieve before writing so the current message cannot retrieve itself.
        context = self.memory_store.search(query, top_k=top_k) if build_context else []

        gate = self.memory_gate.evaluate(query)

        # Conservative v3 policy: sensitive PII can be used in the current
        # interaction but is never written to long-term memory by default.
        allow_persistence = safety.level == SafetyLevel.SAFE or keep_contact

        # Goals are available to the context builder immediately, while the
        # current message still has not been persisted as long-term memory.
        created_goals = self.goal_engine.ingest(query) if allow_persistence else []

        context_packet = None
        if build_context:
            context_packet = self.context_builder.build(
                query,
                memory_candidates=context,
                recent_turns=recent_turns,
                max_tokens=context_budget_tokens,
            )

        stored_ids = self._store(query, said_at, metadata) if allow_persistence else []

        return CognitiveDecision(
            input_text=text,
            safety=safety,
            gate=gate,
            stored_memory_id=stored_ids[0] if stored_ids else None,
            stored_memory_ids=stored_ids,
            created_goal_ids=[goal.id for goal in created_goals],
            context=context,
            context_packet=context_packet,
        )

    def recall(
        self,
        query: str,
        top_k: int = 5,
        *,
        context_budget_tokens: int = 1200,
        recent_turns: Optional[Sequence[ConversationTurn]] = None,
    ) -> ContextPacket:
        """Build the reference context for a query without storing or forgetting anything."""
        safety = self.safety_filter.inspect(query)
        if safety.level == SafetyLevel.FORBIDDEN:
            return ContextPacket(query="", rendered="", estimated_tokens=0,
                                 budget_tokens=context_budget_tokens,
                                 diagnostics={"reason": "forbidden_input"})
        text = safety.redacted_text if safety.redacted_text else query
        candidates = self.memory_store.search(text, top_k=top_k)
        return self.context_builder.build(text, memory_candidates=candidates,
                                          recent_turns=recent_turns, max_tokens=context_budget_tokens)

    def _store(self, text: str, said_at: Optional[datetime] = None, extra: Optional[dict] = None) -> List[str]:
        """Store the message if it states something; a message that only asks is skipped.

        Real messages mix both ("I manage mild asthma. Should I bring the
        inhaler?"), so a question no longer hides the statements around it.
        The message is kept whole: a fact and its subject often sit in
        different sentences.
        """
        gate = self.memory_gate.evaluate(text)
        if not gate.should_store:
            statements = [s for s in split_sentences(text) if not is_question(s)]
            if statements:
                declarative = self.memory_gate.evaluate(
                    " ".join(s.replace("?", ".").replace("¿", "") for s in statements))
                if declarative.should_store:
                    gate = declarative
        if not gate.should_store:
            return []
        metadata = {"gate_reasons": gate.reasons, **(extra or {})}
        if said_at is not None:
            metadata["said_at"] = said_at.isoformat()
        record = MemoryRecord(content=text, kind=gate.kind, importance=gate.importance, metadata=metadata)
        return [self.memory_store.add(record).id]

    def _forget(
        self,
        text: str,
        query: str,
        safety: SafetyDecision,
        top_k: int,
        context_budget_tokens: int,
        recent_turns: Optional[Sequence[ConversationTurn]],
        build_context: bool,
        said_at: Optional[datetime],
    ) -> Optional[CognitiveDecision]:
        finder = getattr(self.memory_store, "forget_candidates", None)
        candidates = (finder(query, top_k=self.forget_candidates) if finder
                      else self.memory_store.search(query, top_k=self.forget_candidates))
        plan = self.forget_planner.plan(query, candidates)
        if not plan.is_request:
            return None

        forgotten = self.memory_store.purge(plan.forget_ids) if plan.forget_ids else 0
        stored_ids: List[str] = []
        if safety.level == SafetyLevel.SAFE:
            for statement in plan.remember:
                stored_ids.extend(self._store(statement, said_at))

        context, packet = [], None
        if build_context:
            context = self.memory_store.search(query, top_k=top_k)
            packet = self.context_builder.build(query, memory_candidates=context,
                                                recent_turns=recent_turns, max_tokens=context_budget_tokens)
        return CognitiveDecision(
            input_text=text,
            safety=safety,
            gate=None,
            stored_memory_id=stored_ids[0] if stored_ids else None,
            stored_memory_ids=stored_ids,
            context=context,
            context_packet=packet,
            forget_request=True,
            forgotten_count=forgotten,
        )
