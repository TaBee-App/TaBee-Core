from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FretPosition:
    string_number: Optional[int]
    fret: int

    @property
    def is_rest(self) -> bool:
        return self.string_number is None
