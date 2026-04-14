from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .FretPosition import FretPosition


@dataclass(frozen=True)
class TabNoteAssignment:
    time_s: float
    frequency_hz: float
    confidence: float
    midi_note: Optional[int]
    note_name: Optional[str]
    position: FretPosition
