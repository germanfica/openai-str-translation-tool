# openai_pricing/models.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class PriceEntry:
    input: Optional[float] = None        # USD per 1M tokens
    cached_input: Optional[float] = None # USD per 1M tokens
    output: Optional[float] = None       # USD per 1M tokens

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
