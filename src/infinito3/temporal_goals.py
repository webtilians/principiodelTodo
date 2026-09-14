from datetime import datetime
from typing import Dict, Optional

from .goals import SimpleGoalEngine
from .types import Goal


class TemporalGoalEngine(SimpleGoalEngine):
    """SimpleGoalEngine plus an explicit projection API for cognitive events."""

    def add_structured_goal(
        self,
        description: str,
        *,
        due_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> Goal:
        goal = Goal(
            description=" ".join(description.strip().split()),
            due_at=due_at,
            metadata=dict(metadata or {}),
        )
        goal.metadata.setdefault("source", "cognitive_event")
        self._goals.append(goal)
        return goal
