from __future__ import annotations

from dataclasses import dataclass
from typing import List

import librosa
import numpy as np

from ..domain.PluckCandidate import PluckCandidate
from ..ports.PluckDetector import PluckDetector


@dataclass(frozen=True)
class PluckDetectionConfig:
    """Configuration for bass pluck candidate detection."""

    attack_low_hz: float = 35.0
    attack_high_hz: float = 1800.0
    envelope_smooth_ms: float = 12.0
    novelty_smooth_ms: float = 20.0
    local_threshold_ms: float = 220.0
    min_peak_distance_ms: float = 80.0
    merge_window_ms: float = 140.0
    threshold_std_ratio: float = 1.5
    threshold_offset: float = 0.04
    min_peak_strength: float = 0.20
    keep_close_peak_ratio: float = 1.35
    gap_fill_min_gap_ms: float = 260.0
    gap_fill_edge_guard_ms: float = 70.0
    gap_fill_threshold_ratio: float = 0.72
    envelope_weight: float = 0.65
    spectral_weight: float = 0.35


class BassPluckDetector(PluckDetector):
    """Bass-oriented pluck detector using envelope rise + spectral flux."""

    def __init__(self, config: PluckDetectionConfig) -> None:
        self._cfg = config

    def detect(self, y: np.ndarray, sr: int) -> List[PluckCandidate]:
        if y.size == 0:
            return []

        y_band = self._band_limit(y, sr)
        envelope = self._amplitude_envelope(y_band, sr)
        envelope_rise = self._positive_derivative(envelope)
        spectral_flux = self._spectral_flux(y_band, sr, len(envelope_rise))

        envelope_score = self._normalize(envelope_rise)
        spectral_score = self._normalize(spectral_flux)
        novelty = (
            self._cfg.envelope_weight * envelope_score
            + self._cfg.spectral_weight * spectral_score
        )
        novelty = self._smooth(
            novelty, self._ms_to_samples(self._cfg.novelty_smooth_ms, sr)
        )

        peaks = self._pick_peaks(novelty, sr)

        candidates: List[PluckCandidate] = []
        for peak in peaks:
            candidates.append(
                PluckCandidate(
                    time_s=peak / sr,
                    strength=float(novelty[peak]),
                    envelope_strength=float(envelope_score[peak]),
                    spectral_strength=float(spectral_score[peak]),
                )
            )
        return candidates

    def _band_limit(self, y: np.ndarray, sr: int) -> np.ndarray:
        fft = librosa.stft(y, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        mask = (freqs >= self._cfg.attack_low_hz) & (freqs <= self._cfg.attack_high_hz)
        fft[~mask, :] = 0.0
        return librosa.istft(fft, length=len(y)).astype(np.float32, copy=False)

    def _amplitude_envelope(self, y: np.ndarray, sr: int) -> np.ndarray:
        return self._smooth(
            np.abs(y), self._ms_to_samples(self._cfg.envelope_smooth_ms, sr)
        )

    def _positive_derivative(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        return np.maximum(np.diff(values, prepend=values[0]), 0.0)

    def _spectral_flux(
        self, y: np.ndarray, sr: int, target_len: int
    ) -> np.ndarray:
        hop_length = 256
        spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop_length))
        flux = np.sqrt(np.sum(np.maximum(np.diff(spec, axis=1), 0.0) ** 2, axis=0))

        if flux.size == 0:
            return np.zeros(target_len, dtype=np.float32)

        flux_times = librosa.frames_to_time(
            np.arange(flux.size), sr=sr, hop_length=hop_length
        )
        target_times = np.arange(target_len, dtype=float) / float(sr)
        resampled = np.interp(
            target_times,
            flux_times,
            flux,
            left=flux[0],
            right=flux[-1],
        )
        return resampled.astype(np.float32, copy=False)

    def _pick_peaks(self, novelty: np.ndarray, sr: int) -> np.ndarray:
        if novelty.size == 0:
            return np.asarray([], dtype=int)

        threshold_window = self._ms_to_samples(self._cfg.local_threshold_ms, sr)
        min_distance = self._ms_to_samples(self._cfg.min_peak_distance_ms, sr)
        merge_distance = self._ms_to_samples(self._cfg.merge_window_ms, sr)
        local_mean = self._moving_average(novelty, threshold_window)
        local_sq_mean = self._moving_average(novelty ** 2, threshold_window)
        local_var = np.maximum(local_sq_mean - (local_mean ** 2), 0.0)
        local_std = np.sqrt(local_var)
        threshold = (
            local_mean
            + (local_std * self._cfg.threshold_std_ratio)
            + self._cfg.threshold_offset
        )

        peaks = librosa.util.peak_pick(
            novelty,
            pre_max=max(1, min_distance // 2),
            post_max=max(1, min_distance // 2),
            pre_avg=max(1, threshold_window // 2),
            post_avg=max(1, threshold_window // 2),
            delta=0.0,
            wait=max(1, min_distance),
        )
        if peaks.size == 0:
            return peaks

        viable = peaks[novelty[peaks] >= self._cfg.min_peak_strength]
        primary = viable[novelty[viable] >= threshold[viable]]
        merged = self._merge_close_peaks(primary, novelty, merge_distance)
        return self._recover_gap_peaks(
            merged,
            viable,
            novelty,
            threshold,
            sr,
        )

    def _merge_close_peaks(
        self,
        peaks: np.ndarray,
        novelty: np.ndarray,
        merge_distance: int,
    ) -> np.ndarray:
        if peaks.size <= 1:
            return peaks

        merged: List[int] = [int(peaks[0])]
        for peak in peaks[1:]:
            current_peak = int(peak)
            prev_peak = merged[-1]

            if (current_peak - prev_peak) > merge_distance:
                merged.append(current_peak)
                continue

            prev_strength = float(novelty[prev_peak])
            current_strength = float(novelty[current_peak])

            if current_strength >= (prev_strength * self._cfg.keep_close_peak_ratio):
                merged[-1] = current_peak

        return np.asarray(merged, dtype=int)

    def _recover_gap_peaks(
        self,
        accepted: np.ndarray,
        viable: np.ndarray,
        novelty: np.ndarray,
        threshold: np.ndarray,
        sr: int,
    ) -> np.ndarray:
        if accepted.size <= 1 or viable.size == 0:
            return accepted

        min_gap = self._ms_to_samples(self._cfg.gap_fill_min_gap_ms, sr)
        edge_guard = self._ms_to_samples(self._cfg.gap_fill_edge_guard_ms, sr)

        recovered = list(int(x) for x in accepted)
        changed = True
        while changed:
            changed = False
            recovered.sort()
            for left, right in zip(recovered, recovered[1:]):
                if (right - left) < min_gap:
                    continue

                interior = viable[(viable > (left + edge_guard)) & (viable < (right - edge_guard))]
                if interior.size == 0:
                    continue

                strength_floor = np.maximum(
                    self._cfg.min_peak_strength,
                    threshold[interior] * self._cfg.gap_fill_threshold_ratio,
                )
                eligible = interior[novelty[interior] >= strength_floor]
                if eligible.size == 0:
                    continue

                best = int(eligible[np.argmax(novelty[eligible])])
                if best not in recovered:
                    recovered.append(best)
                    changed = True
                    break

        return np.asarray(sorted(recovered), dtype=int)

    def _moving_average(self, values: np.ndarray, width: int) -> np.ndarray:
        width = max(1, int(width))
        if values.size == 0:
            return values
        kernel = np.ones(width, dtype=np.float32) / float(width)
        return np.convolve(values, kernel, mode="same")

    def _smooth(self, values: np.ndarray, width: int) -> np.ndarray:
        return self._moving_average(values, width)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values.astype(np.float32, copy=False)
        vmax = float(np.max(values))
        if vmax <= 1e-9:
            return np.zeros_like(values, dtype=np.float32)
        return (values / vmax).astype(np.float32, copy=False)

    def _ms_to_samples(self, ms: float, sr: int) -> int:
        return max(1, int(round((ms / 1000.0) * sr)))
