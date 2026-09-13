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


class SafetyLevel(Enum):
    SAFE = "safe"
    SENSITIVE = "sensitive"
    FORBIDDEN = "forbidden"


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
class CognitiveDecision:
    input_text: str
    safety: SafetyDecision
    gate: Optional[GateDecision]
    stored_memory_id: Optional[str] = None
    created_goal_ids: List[str] = field(default_factory=list)
    context: List[MemoryRecord] = field(default_factory=list)

    @property
    def stored(self) -> bool:
        return self.stored_memory_id is not None
