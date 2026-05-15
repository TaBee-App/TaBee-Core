from __future__ import annotations

from typing import Optional

from .FretPosition import FretPosition
from ..ports.Fretboard import Fretboard


class BassFretboard(Fretboard):
    """
    Standard 4-string bass fretboard.

    String numbers are 1..4 from highest pitched string to lowest:
    1 -> G2 (MIDI 43)
    2 -> D2 (MIDI 38)
    3 -> A1 (MIDI 33)
    4 -> E1 (MIDI 28)
    """

    _TUNINGS = {
        "EADG": {
            1: 43,  # G2
            2: 38,  # D2
            3: 33,  # A1
            4: 28,  # E1
        },
        "BEADG": {
            1: 43,  # G2
            2: 38,  # D2
            3: 33,  # A1
            4: 28,  # E1
            5: 23,  # B0
        },
    }

    def __init__(self, max_frets: int = 24, tuning: str = "EADG"):
        tuning_key = tuning.upper()
        if tuning_key not in self._TUNINGS:
            supported = ", ".join(sorted(self._TUNINGS))
            raise ValueError(f"Unsupported bass tuning '{tuning}'. Supported tunings: {supported}")
        self._tuning = tuning_key
        self._string_tunings = dict(self._TUNINGS[tuning_key])
        self._max_frets = int(max_frets)

    @property
    def tuning(self) -> str:
        return self._tuning

    @property
    def string_numbers(self) -> list[int]:
        return sorted(self._string_tunings)

    def get_candidates(self, target_midi: Optional[int]) -> list[FretPosition]:
        """
        Return every playable (string, fret) position for a MIDI note.

        `None` is treated as a rest so downstream optimization can continue.
        """
        if target_midi is None:
            return [FretPosition(string_number=None, fret=0)]

        candidates: list[FretPosition] = []
        for string_number, open_string_midi in self._string_tunings.items():
            fret = int(target_midi) - open_string_midi
            if 0 <= fret <= self._max_frets:
                candidates.append(FretPosition(string_number=string_number, fret=fret))

        if not candidates:
            return [FretPosition(string_number=None, fret=0)]

        return candidates
