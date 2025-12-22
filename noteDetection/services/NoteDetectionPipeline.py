from typing import List
import numpy as np

from ..domain.DetectedNote import DetectedNote
from ..domain.DetectionResult import DetectionResult


class NoteDetectionPipeline:

    def detect(self, y: np.ndarray, sr: int) -> DetectionResult:
        y = self._preprocess(y)

        onset_times = self._detect_onsets(y, sr)
        pitches = self._estimate_pitch_hz(y, sr)
        tempo = self._estimate_tempo_bpm(y, sr)
        notes = self._build_notes(onset_times, pitches)

        return DetectionResult(
            tempo_bpm=tempo,
            onset_times_s=onset_times,
            pitch_hz=pitches,
            notes=notes,
        )

    def _preprocess(self, y: np.ndarray) -> np.ndarray:
        # Normalize to [-1, 1] safely. Add trimming/noise reduction later if needed.
        max_abs = float(np.max(np.abs(y))) if y.size else 0.0
        if max_abs < 1e-9:
            return y
        return (y / max_abs).astype(np.float32)

    def _detect_onsets(self, y: np.ndarray, sr: int) -> List[float]:
        """
        TODO (implementation choices):
          - librosa.onset.onset_detect -> frames -> times
          - or librosa.onset.onset_strength + peak picking
        Return: onset times in seconds.
        """
        return []

    def _estimate_pitch_hz(self, y: np.ndarray, sr: int) -> List[float]:
        """
        TODO (implementation choices):
          - librosa.yin / librosa.pyin
          - CREPE (later)
        Return: pitch estimate per onset (Hz). Keep aligned with onset list length.
        """
        return []

    def _estimate_tempo_bpm(self, y: np.ndarray, sr: int) -> int:
        """
        TODO:
          - librosa.beat.tempo (returns array)
        """
        return 120

    def _build_notes(self, onsets: List[float], pitches: List[float]) -> List[DetectedNote]:
        notes: List[DetectedNote] = []

        for i, (t, f) in enumerate(zip(onsets, pitches)):
            conf = self._confidence_for_pair(t, f, i)
            notes.append(DetectedNote(time_s=t, frequency_hz=f, confidence=conf))

        return notes

    def _confidence_for_pair(self, time_s: float, freq_hz: float, index: int) -> float:
        # Placeholder heuristic. Replace with something real later.
        if freq_hz <= 0:
            return 0.0
        return 1.0
