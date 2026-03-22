from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np

from ..domain.DetectedNote import DetectedNote
from ..domain.DetectionResult import DetectionResult
from .BassPluckDetector import BassPluckDetector, PluckDetectionConfig
from .OnsetPitchEstimator import OnsetPitchEstimator, PitchConfig


@dataclass(frozen=True)
class TempoConfig:
    """Configuration for tempo estimation from inter-onset intervals (IOI)."""

    preferred_range_bpm: Tuple[float, float] = (60.0, 120.0)


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for basic audio preprocessing."""

    trim_top_db: float = 30.0
    normalize_eps: float = 1e-9


class NoteDetectionPipeline:
    """
    Audio note detection pipeline:
      - trims silence + normalizes
      - detects bass pluck candidates
      - estimates pitch (Hz) and confidence per pluck using librosa.pyin
      - estimates tempo from onset spacing
      - builds domain DetectedNote objects and DetectionResult
    """

    def __init__(
        self,
        *,
        preprocess: Optional[PreprocessConfig] = None,
        pluck: Optional[PluckDetectionConfig] = None,
        pitch: Optional[PitchConfig] = None,
        tempo: Optional[TempoConfig] = None,
    ) -> None:
        self._pre_cfg = preprocess or PreprocessConfig()
        self._tempo_cfg = tempo or TempoConfig()
        self._pluck_detector = BassPluckDetector(pluck or PluckDetectionConfig())
        self._pitch_estimator = OnsetPitchEstimator(pitch or PitchConfig())

    def detect(self, y: np.ndarray, sr: int) -> DetectionResult:
        y = self._preprocess(y)

        pluck_candidates = self._pluck_detector.detect(y, sr)
        onset_times = [candidate.time_s for candidate in pluck_candidates]
        pitches, confidences = self._pitch_estimator.estimate(
            y, sr, pluck_candidates
        )
        tempo_bpm = self._estimate_tempo_bpm_from_onsets(onset_times)
        notes = self._build_notes(onset_times, pitches, confidences)

        return DetectionResult(
            tempo_bpm=tempo_bpm,
            onset_times_s=onset_times,
            pitch_hz=pitches,
            notes=notes,
            pluck_candidates=pluck_candidates,
        )

    def _preprocess(self, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y)
        if y.size == 0:
            return y.astype(np.float32)

        y, _ = librosa.effects.trim(y, top_db=self._pre_cfg.trim_top_db)

        max_abs = float(np.max(np.abs(y))) if y.size else 0.0
        if max_abs > self._pre_cfg.normalize_eps:
            y = y / max_abs

        return y.astype(np.float32, copy=False)

    def _estimate_tempo_bpm_from_onsets(self, onset_times: List[float]) -> int:
        if len(onset_times) < 2:
            return 0

        ioi = np.diff(np.asarray(onset_times, dtype=float))
        ioi = ioi[(ioi > 1e-3) & np.isfinite(ioi)]
        if ioi.size == 0:
            return 0

        base = float(np.median(ioi))
        bpm = 60.0 / base

        lo, hi = self._tempo_cfg.preferred_range_bpm
        while bpm > hi:
            bpm /= 2.0
        while bpm < lo:
            bpm *= 2.0

        return int(round(bpm))

    def _build_notes(
        self,
        onset_times: List[float],
        pitches: List[float],
        confidences: List[float],
    ) -> List[DetectedNote]:
        return [
            DetectedNote(time_s=float(t), frequency_hz=float(f), confidence=float(c))
            for t, f, c in zip(onset_times, pitches, confidences)
        ]
