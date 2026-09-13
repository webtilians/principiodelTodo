from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class MemoryKind(Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    USER_MODEL = "user_model"


class MemoryStatus(Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FORGOTTEN = "forgotten"


class SafetyLevel(Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    FORBIDDEN = "forbidden"


class ContextSource(Enum):
    GOAL = "goal"
    USER_MODEL = "user_model"
    MEMORY = "memory"
    RECENT = "recent"


@dataclass
class SafetyDecision:
    level: SafetyLevel
    reason: str = ""
    redacted_text: Optional[str] = None


@dataclass
class MemoryRecord:
    content: str
    kind: MemoryKind
    importance: float
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0
    status: MemoryStatus = MemoryStatus.ACTIVE
    supersedes_id: Optional[str] = None
    fact_subject: Optional[str] = None
    fact_predicate: Optional[str] = None
    fact_value: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class GateDecision:
    should_store: bool
    importance: float
    kind: MemoryKind
    reasons: List[str] = field(default_factory=list)


@dataclass
class Goal:
    description: str
    due_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConversationTurn:
    role: str
    content: str


@dataclass
class ContextItem:
    source: ContextSource
    content: str
    score: float
    estimated_tokens: int
    memory_id: Optional[str] = None
    goal_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextPacket:
    query: str
    rendered: str
    estimated_tokens: int
    budget_tokens: int
    items: List[ContextItem] = field(default_factory=list)
    dropped_count: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaintenanceReport:
    consolidated: int = 0
    forgotten: int = 0
    restored: int = 0


@dataclass
class CognitiveDecision:
    input_text: str
    safety: SafetyDecision
    gate: Optional[GateDecision]
    stored_memory_id: Optional[str] = None
    created_goal_ids: List[str] = field(default_factory=list)
    context: List[MemoryRecord] = field(default_factory=list)
    context_packet: Optional[ContextPacket] = None

    @property
    def stored(self) -> bool:
        return self.stored_memory_id is not None
