from __future__ import annotations

from typing import Optional, Sequence

from music.pitch.PitchConverter import PitchConverter
from noteDetection.domain.DetectedNote import DetectedNote
from noteDetection.domain.DetectionResult import DetectionResult

from ..domain.BassFretboard import BassFretboard
from ..domain.FretPosition import FretPosition
from ..domain.TabNoteAssignment import TabNoteAssignment
from ..ports.Fretboard import Fretboard
from ..ports.TabOptimizer import TabOptimizer
from .PlayabilityOptimizer import PlayabilityOptimizer


class TabGenerationService:
    """
    Converts detected note events into playable bass fretboard positions.

    This layer intentionally sits after pitch detection and note labeling.
    It does not modify the detection pipeline.
    """

    def __init__(
        self,
        fretboard: Optional[Fretboard] = None,
        optimizer: Optional[TabOptimizer] = None,
        bass_octave_threshold_midi: int = 45,
    ) -> None:
        self._fretboard = fretboard or BassFretboard()
        self._optimizer = optimizer or PlayabilityOptimizer()
        self._bass_octave_threshold_midi = int(bass_octave_threshold_midi)

    def generate_from_detection_result(
        self,
        result: DetectionResult,
    ) -> list[TabNoteAssignment]:
        return self.generate_from_detected_notes(result.notes)

    def generate_from_detected_notes(
        self,
        notes: Sequence[DetectedNote],
    ) -> list[TabNoteAssignment]:
        candidates_per_note: list[list[FretPosition]] = []
        normalized_midi: list[Optional[int]] = []

        for note in notes:
            midi_note = self._normalize_bass_midi(
                PitchConverter.hz_to_midi(note.frequency_hz)
            )
            normalized_midi.append(midi_note)
            candidates_per_note.append(self._fretboard.get_candidates(midi_note))

        optimal_path = self._optimizer.optimize(candidates_per_note)

        assignments: list[TabNoteAssignment] = []
        for note, midi_note, position in zip(notes, normalized_midi, optimal_path):
            note_name = self._midi_to_name(midi_note)
            assignments.append(
                TabNoteAssignment(
                    time_s=note.time_s,
                    frequency_hz=note.frequency_hz,
                    confidence=note.confidence,
                    midi_note=midi_note,
                    note_name=note_name,
                    position=position,
                )
            )

        return assignments

    def _normalize_bass_midi(self, midi_note: Optional[int]) -> Optional[int]:
        if midi_note is None:
            return None

        corrected = int(midi_note)
        while corrected > self._bass_octave_threshold_midi:
            corrected -= 12
        return corrected

    def _midi_to_name(self, midi_note: Optional[int]) -> Optional[str]:
        if midi_note is None:
            return None
        name, octave = PitchConverter.midi_to_name_octave(midi_note)
        return f"{name}{octave}"
