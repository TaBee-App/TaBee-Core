from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import librosa
import numpy as np

from ..domain.DetectedNote import DetectedNote
from ..domain.DetectionResult import DetectionResult


@dataclass(frozen=True)
class PitchConfig:
    """Configuration for f0 tracking around detected onsets."""

    fmin_hz: float = 32.0
    fmax_hz: float = 120.0
    hop_length: int = 256
    frame_length: int = 8192
    start_after_onset_s: float = 0.06
    max_window_s: float = 0.22
    pre_next_onset_margin_s: float = 0.02
    min_confidence: float = 0.01


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
      - detects onset times
      - estimates pitch (Hz) and confidence per onset using librosa.pyin
      - estimates tempo from onset spacing
      - builds domain DetectedNote objects and DetectionResult
    """

    def __init__(
        self,
        *,
        preprocess: Optional[PreprocessConfig] = None,
        pitch: Optional[PitchConfig] = None,
        tempo: Optional[TempoConfig] = None,
    ) -> None:
        """
        Create a pipeline with optional configuration overrides.

        Args:
            preprocess: Parameters for trimming/normalization.
            pitch: Parameters for pitch tracking around onsets.
            tempo: Parameters for IOI-based tempo estimation.
        """
        self._pre_cfg = preprocess or PreprocessConfig()
        self._pitch_cfg = pitch or PitchConfig()
        self._tempo_cfg = tempo or TempoConfig()

    def detect(self, y: np.ndarray, sr: int) -> DetectionResult:
        """
        Run full detection on an audio buffer.

        Args:
            y: Mono audio samples (float or int). If stereo is provided upstream, convert to mono first.
            sr: Sample rate.

        Returns:
            DetectionResult containing tempo (bpm), onset times (s), raw pitch list (Hz),
            and DetectedNote objects (time, frequency, confidence).
        """
        y = self._preprocess(y)

        onset_times = self._detect_onsets(y, sr)

        pitches, confidences = self._estimate_pitch_hz_and_confidence(
            y, sr, onset_times
        )

        tempo_bpm = self._estimate_tempo_bpm_from_onsets(onset_times)

        notes = self._build_notes(onset_times, pitches, confidences)

        return DetectionResult(
            tempo_bpm=tempo_bpm,
            onset_times_s=onset_times,
            pitch_hz=pitches,
            notes=notes,
        )

    def _preprocess(self, y: np.ndarray) -> np.ndarray:
        """
        Trim leading/trailing silence and normalize amplitude.

        Notes:
            - Uses librosa.effects.trim for silence removal.
            - Normalizes by max abs value to keep y in [-1, 1] (if non-silent).

        Args:
            y: Audio samples.

        Returns:
            Preprocessed mono float32 audio.
        """
        y = np.asarray(y)
        if y.size == 0:
            return y.astype(np.float32)

        y, _ = librosa.effects.trim(y, top_db=self._pre_cfg.trim_top_db)

        max_abs = float(np.max(np.abs(y))) if y.size else 0.0
        if max_abs > self._pre_cfg.normalize_eps:
            y = y / max_abs

        return y.astype(np.float32, copy=False)

    def _detect_onsets(self, y: np.ndarray, sr: int) -> List[float]:
        """
        Detect note attack times (onsets) in seconds.

        Uses librosa's onset strength + peak picking via onset_detect.

        Args:
            y: Preprocessed audio.
            sr: Sample rate.

        Returns:
            Sorted onset times in seconds.
        """
        onset_frames = librosa.onset.onset_detect(
            y=y,
            sr=sr,
            units="frames",
            backtrack=False,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=0.2,
            wait=10,
        )
        return librosa.frames_to_time(onset_frames, sr=sr).tolist()

    def _estimate_pitch_hz_and_confidence(
        self,
        y: np.ndarray,
        sr: int,
        onset_times: List[float],
    ) -> Tuple[List[float], List[float]]:
        """
        Estimate pitch (Hz) per onset with a confidence score.

        Strategy:
            1) Run librosa.pyin once over the whole signal to get f0(t) and voiced_prob(t).
            2) For each onset, sample a window *after* the attack transient.
            3) Use median f0 and median voiced_prob in that window.

        Confidence:
            - Uses pyin's voiced probability (0..1).
            - Values can be small for bass/distorted signals; treat as relative confidence.

        Args:
            y: Preprocessed audio.
            sr: Sample rate.
            onset_times: Onset times in seconds.

        Returns:
            (pitches_hz, confidences) aligned with onset_times.
            Unvoiced/invalid entries are returned as 0.0 pitch and 0.0 confidence.
        """
        if not onset_times:
            return [], []

        cfg = self._pitch_cfg

        # Run pYIN once (expensive); reuse for all onsets
        f0, _voiced_flag, voiced_prob = librosa.pyin(
            y=y,
            fmin=cfg.fmin_hz,
            fmax=cfg.fmax_hz,
            sr=sr,
            frame_length=cfg.frame_length,
            hop_length=cfg.hop_length,
        )

        times = librosa.times_like(f0, sr=sr, hop_length=cfg.hop_length)

        pitches: List[float] = []
        confidences: List[float] = []

        n = len(onset_times)

        for i, onset_t in enumerate(onset_times):
            next_onset_t = (
                onset_times[i + 1] if i + 1 < n else (onset_t + cfg.max_window_s)
            )

            # Window after onset, bounded by next onset
            t0 = onset_t + cfg.start_after_onset_s
            t1 = min(
                onset_t + cfg.max_window_s, next_onset_t - cfg.pre_next_onset_margin_s
            )

            # If riff is very dense, guarantee a minimal window
            if t1 <= t0:
                t1 = t0 + 0.06

            i0 = int(np.searchsorted(times, t0, side="left"))
            i1 = int(np.searchsorted(times, t1, side="right"))

            # Clamp and ensure non-empty slice
            if len(times) == 0:
                pitches.append(0.0)
                confidences.append(0.0)
                continue

            i0 = max(0, min(i0, len(times) - 1))
            i1 = max(i0 + 1, min(i1, len(times)))

            f_slice = f0[i0:i1]
            p_slice = voiced_prob[i0:i1]

            mask = np.isfinite(f_slice)
            if not np.any(mask):
                pitches.append(0.0)
                confidences.append(0.0)
                continue

            f_valid = f_slice[mask]
            p_valid = p_slice[mask]

            pitch = float(np.median(f_valid))
            conf = float(np.median(p_valid))

            if (not np.isfinite(pitch)) or pitch <= 0.0 or conf < cfg.min_confidence:
                pitch = 0.0
                conf = 0.0

            pitches.append(pitch)
            confidences.append(conf)

        return pitches, confidences

    def _estimate_tempo_bpm_from_onsets(self, onset_times: List[float]) -> int:
        """
        Estimate tempo from inter-onset intervals (IOI), then fold to a preferred BPM range.

        Beat trackers often return half/double tempo on short monophonic riffs.
        This method:
          - takes the median IOI
          - converts to BPM assuming each onset is a beat
          - repeatedly halves/doubles into a preferred range (default 60..120)

        Args:
            onset_times: Onset times in seconds.

        Returns:
            Estimated tempo in BPM (integer). Returns 0 if insufficient data.
        """
        if len(onset_times) < 2:
            return 0

        ioi = np.diff(np.asarray(onset_times, dtype=float))
        ioi = ioi[(ioi > 1e-3) & np.isfinite(ioi)]
        if ioi.size == 0:
            return 0

        base = float(np.median(ioi))  # seconds per onset
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
        """
        Convert aligned onset/pitch/confidence arrays into DetectedNote objects.

        Args:
            onset_times: Onset times in seconds.
            pitches: Pitch estimates in Hz (0.0 if unvoiced).
            confidences: Confidence scores in [0..1] (0.0 if unvoiced).

        Returns:
            List of DetectedNote objects aligned with inputs.
        """
        return [
            DetectedNote(time_s=float(t), frequency_hz=float(f), confidence=float(c))
            for t, f, c in zip(onset_times, pitches, confidences)
        ]
