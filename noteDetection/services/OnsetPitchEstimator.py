from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import librosa
import numpy as np

from ..domain.PluckCandidate import PluckCandidate
from ..ports.PitchEstimator import PitchEstimator


@dataclass(frozen=True)
class PitchConfig:
    """Configuration for bass pitch readout after pluck candidates."""

    fmin_hz: float = 32.0
    fmax_hz: float = 220.0
    hop_length: int = 256
    frame_length: int = 8192
    min_read_delay_s: float = 0.035
    max_read_delay_s: float = 0.11
    max_window_s: float = 0.22
    pre_next_pluck_margin_s: float = 0.015
    min_window_s: float = 0.06
    min_confidence: float = 0.01
    harmonic_tolerance_semitones: float = 0.45
    max_note_jump_semitones: int = 9
    short_note_delay_ratio: float = 0.28
    short_note_window_ratio: float = 0.65


class OnsetPitchEstimator(PitchEstimator):
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

            note_span_s = max(0.0, next_onset_t - onset_t)
            delay_s = self._read_delay_s(candidate, note_span_s)
            t0 = onset_t + delay_s
            max_window_s = min(cfg.max_window_s, max(cfg.min_window_s, note_span_s * cfg.short_note_window_ratio))
            t1 = min(
                onset_t + delay_s + max_window_s,
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

            f_valid = f_slice[mask].astype(float)
            p_valid = p_slice[mask].astype(float)

            pitch = self._robust_pitch(f_valid, p_valid)
            pitch = self._stabilize_against_previous(pitch, pitches)
            conf = float(np.median(p_valid))
            if (not np.isfinite(pitch)) or pitch <= 0.0 or conf < cfg.min_confidence:
                pitch = 0.0
                conf = 0.0

            pitches.append(pitch)
            confidences.append(conf)

        return pitches, confidences

    def _read_delay_s(self, candidate: PluckCandidate, note_span_s: float) -> float:
        env = max(0.0, min(candidate.envelope_strength, 1.0))
        attack_weight = 1.0 - env
        span = self._cfg.max_read_delay_s - self._cfg.min_read_delay_s
        delay = self._cfg.min_read_delay_s + (attack_weight * span)
        if note_span_s > 0.0:
            delay = min(delay, max(0.012, note_span_s * self._cfg.short_note_delay_ratio))
        return delay

    def _robust_pitch(self, frequencies: np.ndarray, probabilities: np.ndarray) -> float:
        if frequencies.size == 0:
            return 0.0

        probabilities = np.maximum(probabilities, 1e-6)
        midi = 69.0 + (12.0 * np.log2(frequencies / 440.0))
        rounded = np.rint(midi).astype(int)
        unique_notes = np.unique(rounded)

        best_note = int(unique_notes[0])
        best_weight = -1.0
        for note in unique_notes:
            distance = np.abs(midi - float(note))
            in_cluster = distance <= self._cfg.harmonic_tolerance_semitones
            weight = float(np.sum(probabilities[in_cluster]))
            if weight > best_weight:
                best_weight = weight
                best_note = int(note)

        note_mask = np.abs(midi - float(best_note)) <= self._cfg.harmonic_tolerance_semitones
        if not np.any(note_mask):
            return float(np.median(frequencies))

        return self._weighted_median(frequencies[note_mask], probabilities[note_mask])

    def _stabilize_against_previous(self, pitch: float, previous_pitches: List[float]) -> float:
        previous = self._last_valid_pitch(previous_pitches)
        if previous is None or pitch <= 0.0:
            return pitch

        corrected = float(pitch)
        previous_midi = 69.0 + (12.0 * np.log2(previous / 440.0))

        while corrected > self._cfg.fmin_hz * 2.0:
            current_midi = 69.0 + (12.0 * np.log2(corrected / 440.0))
            if abs(current_midi - previous_midi) <= self._cfg.max_note_jump_semitones:
                break

            octave_down = corrected / 2.0
            octave_midi = 69.0 + (12.0 * np.log2(octave_down / 440.0))
            if abs(octave_midi - previous_midi) >= abs(current_midi - previous_midi):
                break
            corrected = octave_down

        return corrected

    def _last_valid_pitch(self, pitches: List[float]) -> Optional[float]:
        for pitch in reversed(pitches):
            if pitch > 0.0 and np.isfinite(pitch):
                return float(pitch)
        return None

    def _weighted_median(self, values: np.ndarray, weights: np.ndarray) -> float:
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cumulative = np.cumsum(sorted_weights)
        midpoint = float(cumulative[-1]) / 2.0
        index = int(np.searchsorted(cumulative, midpoint, side="left"))
        return float(sorted_values[min(index, sorted_values.size - 1)])
