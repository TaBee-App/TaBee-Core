from typing import List

import librosa
import numpy as np

from ..domain.DetectedNote import DetectedNote
from ..domain.DetectionResult import DetectionResult


class NoteDetectionPipeline:

    def detect(self, y, sr):
        y = self._preprocess(y)

        onset_times = self._detect_onsets(y, sr)
        pitches = self._estimate_pitch_hz(y, sr, onset_times)  # <-- FIX
        tempo = self._estimate_tempo_bpm(y, sr)
        notes = self._build_notes(onset_times, pitches)

        return DetectionResult(
            tempo_bpm=tempo,
            onset_times_s=onset_times,
            pitch_hz=pitches,
            notes=notes,
        )

    def _preprocess(self, y: np.ndarray) -> np.ndarray:
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=30)

        # Normalize safely
        max_abs = np.max(np.abs(y)) if y.size else 0.0
        if max_abs > 1e-9:
            y = y / max_abs

        return y.astype(np.float32)

    def _detect_onsets(self, y: np.ndarray, sr: int) -> List[float]:
        """
        Uses librosa onset strength + peak picking.
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

        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        return onset_times.tolist()

    def _estimate_pitch_hz(
        self,
        y: np.ndarray,
        sr: int,
        onset_times: List[float],
    ) -> List[float]:
        """
        Pitch per onset using librosa.yin.
        Strategy:
          - Run YIN once
          - Sample pitch around each onset
        """
        if not onset_times:
            return []

        # Run YIN over whole signal
        f0 = librosa.yin(
            y=y,
            fmin=librosa.note_to_hz("E1"),
            fmax=librosa.note_to_hz("C5"),
            sr=sr,
        )

        times = librosa.times_like(f0, sr=sr)

        pitches: List[float] = []

        """
        Yapılan değişiklik : Onset zamanından 20ms sonra başlayıp 50ms süren bir pencere açarak, 
        bu pencere içindeki geçerli (pozitif) pitch değerlerinin medyanını alıyoruz. Eğer bu pencerede 
        geçerli bir pitch yoksa, o zaman 0 Hz olarak kabul ediyoruz. Bu sayede, onset zamanında anlık 
        bir pitch değeri yerine, onset sonrası kısa bir süre boyunca gözlemlenen pitch değerlerinin 
        daha stabil bir temsilini elde ediyoruz. Bu yöntem, özellikle gürültülü sinyallerde veya hızlı 
        geçişlerde daha güvenilir sonuçlar verebilir.
        """
        window_start_offset = 0.02  
        window_duration = 0.05      

        for onset_t in onset_times:
            start_t = onset_t + window_start_offset
            end_t = start_t + window_duration
            
            valid_indices = np.where((times >= start_t) & (times <= end_t))[0]
            
            if len(valid_indices) == 0:
                idx = np.argmin(np.abs(times - onset_t))
                valid_indices = [idx]
                
            window_pitches = f0[valid_indices]
            valid_pitches = window_pitches[window_pitches > 0]
            
            if len(valid_pitches) > 0:
                pitch = float(np.median(valid_pitches))
            else:
                pitch = 0.0
                
            pitches.append(pitch)

        return pitches

    def _estimate_tempo_bpm(self, y: np.ndarray, sr: int) -> int:
        tempo = librosa.beat.tempo(y=y, sr=sr)
        if tempo.size == 0:
            return 0
        return int(round(float(tempo[0])))

    def _build_notes(
        self,
        onset_times: List[float],
        pitches: List[float],
    ) -> List[DetectedNote]:

        notes: List[DetectedNote] = []

        for i, (t, f) in enumerate(zip(onset_times, pitches)):
            confidence = self._estimate_confidence(f)
            notes.append(
                DetectedNote(
                    time_s=t,
                    frequency_hz=f,
                    confidence=confidence,
                )
            )

        return notes


    def _estimate_confidence(self, frequency_hz: float) -> float:
        """
        Very naive confidence:
          - 0 if no pitch
          - 0.7–1.0 otherwise
        """
        if frequency_hz <= 0:
            return 0.0
        return 0.8
