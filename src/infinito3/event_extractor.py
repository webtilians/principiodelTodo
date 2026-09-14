import re
from typing import Optional

from .cognitive_events import RuleBasedCognitiveEventExtractor


class TemporalCognitiveEventExtractor(RuleBasedCognitiveEventExtractor):
    """Production event extractor wrapper with conservative value cleanup.

    Surface phrases such as `I moved to Bilbao last month` carry temporal
    framing that must not become part of the structured value. The cleanup is
    generic and leaves the original source text intact in event provenance.
    """

    @classmethod
    def _clean_value(cls, value: Optional[str]) -> str:
        cleaned = super()._clean_value(value)
        cleaned = re.sub(
            r"\s+(?:last|this)\s+(?:week|month|year)$",
            "",
            cleaned,
            flags=re.I,
        )
        cleaned = re.sub(
            r"\s+(?:where i live now|where i am living now)$",
            "",
            cleaned,
            flags=re.I,
        )
        return cleaned.strip()
