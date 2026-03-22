from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np

from ..domain.PluckCandidate import PluckCandidate


@dataclass(frozen=True)
class PitchConfig:
    """Configuration for bass pitch readout after pluck candidates."""

    fmin_hz: float = 32.0
    fmax_hz: float = 120.0
    hop_length: int = 256
    frame_length: int = 8192
    min_read_delay_s: float = 0.02
    max_read_delay_s: float = 0.08
    max_window_s: float = 0.18
    pre_next_pluck_margin_s: float = 0.015
    min_window_s: float = 0.05
    min_confidence: float = 0.01


class OnsetPitchEstimator:
    """Reads stable bass pitch after pluck candidates."""

    def __init__(self, config: PitchConfig) -> None:
        self._cfg = config

    def estimate(
        self, y: np.ndarray, sr: int, candidates: List[PluckCandidate]
    ) -> Tuple[List[float], List[float]]:
        if not candidates:
            return [], []

        cfg = self._cfg
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

        n = len(candidates)
        for i, candidate in enumerate(candidates):
            onset_t = candidate.time_s
            next_onset_t = (
                candidates[i + 1].time_s
                if i + 1 < n
                else onset_t + cfg.max_window_s + cfg.max_read_delay_s
            )

            delay_s = self._read_delay_s(candidate)
            t0 = onset_t + delay_s
            t1 = min(
                onset_t + delay_s + cfg.max_window_s,
                next_onset_t - cfg.pre_next_pluck_margin_s,
            )

            if t1 <= t0:
                t1 = t0 + cfg.min_window_s

            i0 = int(np.searchsorted(times, t0, side="left"))
            i1 = int(np.searchsorted(times, t1, side="right"))

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

    def _read_delay_s(self, candidate: PluckCandidate) -> float:
        env = max(0.0, min(candidate.envelope_strength, 1.0))
        attack_weight = 1.0 - env
        span = self._cfg.max_read_delay_s - self._cfg.min_read_delay_s
        return self._cfg.min_read_delay_s + (attack_weight * span)
