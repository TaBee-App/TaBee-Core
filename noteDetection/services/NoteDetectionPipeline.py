from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import warnings

import librosa
import numpy as np

from ..domain.DetectedNote import DetectedNote
from ..domain.DetectionResult import DetectionResult
from ..ports.PitchEstimator import PitchEstimator
from ..ports.PluckDetector import PluckDetector
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
        pluck_detector: Optional[PluckDetector] = None,
        pitch: Optional[PitchConfig] = None,
        pitch_estimator: Optional[PitchEstimator] = None,
        tempo: Optional[TempoConfig] = None,
    ) -> None:
        self._pre_cfg = preprocess or PreprocessConfig()
        self._tempo_cfg = tempo or TempoConfig()
        self._pluck_detector = pluck_detector or BassPluckDetector(
            pluck or PluckDetectionConfig()
        )
        self._pitch_estimator = pitch_estimator or OnsetPitchEstimator(
            pitch or PitchConfig()
        )

    def detect(self, y: np.ndarray, sr: int) -> DetectionResult:
        y, trim_offset_s = self._preprocess(y, sr)

        pluck_candidates = self._pluck_detector.detect(y, sr)
        relative_onset_times = [candidate.time_s for candidate in pluck_candidates]
        onset_times = [time_s + trim_offset_s for time_s in relative_onset_times]
        pitches, confidences = self._pitch_estimator.estimate(
            y, sr, pluck_candidates
        )
        tempo_bpm = self._estimate_tempo_bpm(y, sr, relative_onset_times)
        durations = self._estimate_note_durations(y, sr, relative_onset_times)
        notes = self._build_notes(onset_times, pitches, confidences, durations)

        return DetectionResult(
            tempo_bpm=tempo_bpm,
            onset_times_s=onset_times,
            pitch_hz=pitches,
            notes=notes,
            pluck_candidates=pluck_candidates,
        )

    def _preprocess(self, y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
        y = np.asarray(y)
        if y.size == 0:
            return y.astype(np.float32), 0.0

        y, trim_index = librosa.effects.trim(y, top_db=self._pre_cfg.trim_top_db)
        trim_offset_s = float(trim_index[0]) / float(sr) if sr > 0 else 0.0

        max_abs = float(np.max(np.abs(y))) if y.size else 0.0
        if max_abs > self._pre_cfg.normalize_eps:
            y = y / max_abs

        return y.astype(np.float32, copy=False), trim_offset_s

    def _estimate_tempo_bpm(self, y: np.ndarray, sr: int, onset_times: List[float]) -> int:
        tracked = self._estimate_tempo_bpm_with_beat_tracker(y, sr)
        if tracked:
            return tracked
        return self._estimate_tempo_bpm_from_onsets(onset_times)

    def _estimate_tempo_bpm_with_beat_tracker(self, y: np.ndarray, sr: int) -> int:
        if y.size == 0:
            return 0

        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                tempo_curve = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    aggregate=None,
                )
            tempo_values = np.asarray(tempo_curve, dtype=float)
            tempo_values = tempo_values[np.isfinite(tempo_values) & (tempo_values > 0)]
            if tempo_values.size:
                tempo_value = float(np.median(tempo_values))
            else:
                tempo, _beats = librosa.beat.beat_track(
                    onset_envelope=onset_env,
                    sr=sr,
                    units="time",
                )
                tempo_value = float(np.asarray(tempo).reshape(-1)[0])
        except Exception:
            return 0

        if not np.isfinite(tempo_value) or tempo_value <= 0:
            return 0

        lo, hi = self._tempo_cfg.preferred_range_bpm
        while tempo_value > hi * 1.5:
            tempo_value /= 2.0
        while tempo_value < lo / 1.5:
            tempo_value *= 2.0

        return int(round(tempo_value))

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
        durations: List[float],
    ) -> List[DetectedNote]:
        return [
            DetectedNote(
                time_s=float(t),
                frequency_hz=float(f),
                confidence=float(c),
                duration_s=float(d),
            )
            for t, f, c, d in zip(onset_times, pitches, confidences, durations)
        ]

    def _estimate_note_durations(
        self,
        y: np.ndarray,
        sr: int,
        onset_times: List[float],
    ) -> List[float]:
        if not onset_times:
            return []

        envelope = np.abs(y)
        envelope = self._smooth(envelope, max(1, int(round(sr * 0.025))))
        max_env = float(np.max(envelope)) if envelope.size else 0.0
        if max_env > 1e-9:
            envelope = envelope / max_env

        durations: List[float] = []
        onset_samples = [max(0, min(len(y) - 1, int(round(t * sr)))) for t in onset_times]
        hold_samples = max(1, int(round(sr * 0.09)))
        attack_guard = max(1, int(round(sr * 0.08)))
        min_duration = max(1, int(round(sr * 0.08)))

        for index, start in enumerate(onset_samples):
            next_start = onset_samples[index + 1] if index + 1 < len(onset_samples) else len(y)
            search_start = min(next_start, start + attack_guard)
            search_end = max(search_start, next_start)
            segment = envelope[start:search_end]
            local_peak = float(np.max(segment)) if segment.size else 0.0
            threshold = max(0.025, local_peak * 0.18)

            end = next_start
            for sample in range(search_start, max(search_start, search_end - hold_samples)):
                window = envelope[sample:sample + hold_samples]
                if window.size and float(np.max(window)) <= threshold:
                    end = sample
                    break

            end = max(start + min_duration, min(end, next_start))
            durations.append((end - start) / float(sr))

        return durations

    def _smooth(self, values: np.ndarray, width: int) -> np.ndarray:
        width = max(1, int(width))
        if values.size == 0:
            return values
        kernel = np.ones(width, dtype=np.float32) / float(width)
        return np.convolve(values, kernel, mode="same")
