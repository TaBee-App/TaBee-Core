from typing import List

from .DetectedNote import DetectedNote
from .PluckCandidate import PluckCandidate


class DetectionResult:
    def __init__(
        self,
        tempo_bpm: int,
        onset_times_s: List[float],
        pitch_hz: List[float],
        notes: List[DetectedNote],
        pluck_candidates: List[PluckCandidate] | None = None,
    ):
        self._tempo_bpm = int(tempo_bpm)
        self._onset_times_s = list(onset_times_s)
        self._pitch_hz = list(pitch_hz)
        self._notes = list(notes)
        self._pluck_candidates = list(pluck_candidates or [])

    @property
    def tempo_bpm(self) -> int:
        return self._tempo_bpm

    @property
    def onset_times_s(self) -> List[float]:
        return list(self._onset_times_s)

    @property
    def pitch_hz(self) -> List[float]:
        return list(self._pitch_hz)

    @property
    def notes(self) -> List[DetectedNote]:
        return list(self._notes)

    @property
    def pluck_candidates(self) -> List[PluckCandidate]:
        return list(self._pluck_candidates)
