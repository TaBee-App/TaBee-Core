from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from music.pitch.PitchConverter import PitchConverter

from ..domain.DetectedNote import DetectedNote
from ..domain.DetectionResult import DetectionResult


@dataclass(frozen=True)
class NotePostProcessConfig:
    """Timing cleanup for detected bass note events."""

    enabled: bool = True
    quantize_subdivisions_per_beat: int = 4
    min_confidence: float = 0.01
    min_separation_s: float = 0.07
    max_duration_beats: float = 4.0


class NoteEventPostProcessor:
    """
    Cleans up the raw detector output before tab generation.

    The detector can emit several very close events for one pluck, especially in
    long isolated stems. This stage keeps the strongest event per rhythmic grid
    slot and snaps timing to a beat grid so repeated riffs render consistently.
    """

    def __init__(self, config: NotePostProcessConfig | None = None) -> None:
        self._cfg = config or NotePostProcessConfig()

    def process(self, result: DetectionResult) -> DetectionResult:
        if not self._cfg.enabled:
            return result

        notes = self._valid_notes(result.notes)
        notes = self._collapse_close_notes(notes)
        notes = self._quantize_notes(notes, result.tempo_bpm)

        return DetectionResult(
            tempo_bpm=result.tempo_bpm,
            onset_times_s=[note.time_s for note in notes],
            pitch_hz=[note.frequency_hz for note in notes],
            notes=notes,
            pluck_candidates=result.pluck_candidates,
        )

    def _valid_notes(self, notes: Iterable[DetectedNote]) -> list[DetectedNote]:
        return sorted(
            (
                note
                for note in notes
                if note.frequency_hz > 0.0 and note.confidence >= self._cfg.min_confidence
            ),
            key=lambda note: note.time_s,
        )

    def _collapse_close_notes(self, notes: list[DetectedNote]) -> list[DetectedNote]:
        if len(notes) <= 1:
            return notes

        collapsed: list[DetectedNote] = []
        pending = notes[0]

        for note in notes[1:]:
            same_pitch = self._midi(pending) == self._midi(note)
            close = (note.time_s - pending.time_s) < self._cfg.min_separation_s

            if close or same_pitch and (note.time_s - pending.time_s) < self._cfg.min_separation_s * 1.5:
                pending = self._stronger(pending, note)
                continue

            collapsed.append(pending)
            pending = note

        collapsed.append(pending)
        return collapsed

    def _quantize_notes(self, notes: list[DetectedNote], tempo_bpm: int) -> list[DetectedNote]:
        if len(notes) <= 1 or tempo_bpm <= 0:
            return notes

        grid_s = 60.0 / float(tempo_bpm) / max(1, self._cfg.quantize_subdivisions_per_beat)
        grid_items: list[tuple[int, DetectedNote]] = []
        for note in notes:
            grid_index = max(0, int(round(note.time_s / grid_s)))

            if grid_items and grid_index <= grid_items[-1][0]:
                previous_grid, previous_note = grid_items[-1]
                close_duplicate = (note.time_s - previous_note.time_s) < self._cfg.min_separation_s

                if close_duplicate:
                    grid_items[-1] = (previous_grid, self._stronger(previous_note, note))
                    continue

                grid_index = previous_grid + 1

            grid_items.append((grid_index, note))

        quantized: list[DetectedNote] = []
        max_duration_s = self._cfg.max_duration_beats * 60.0 / float(tempo_bpm)

        for index, (grid_index, note) in enumerate(grid_items):
            time_s = grid_index * grid_s
            next_time_s = grid_items[index + 1][0] * grid_s if index + 1 < len(grid_items) else None
            duration_s = note.duration_s

            if duration_s is None or duration_s <= 0:
                duration_s = grid_s

            duration_s = max(grid_s, round(duration_s / grid_s) * grid_s)
            duration_s = min(duration_s, max_duration_s)
            if next_time_s is not None:
                duration_s = min(duration_s, max(grid_s, next_time_s - time_s))

            quantized.append(
                DetectedNote(
                    time_s=time_s,
                    frequency_hz=note.frequency_hz,
                    confidence=note.confidence,
                    duration_s=duration_s,
                )
            )

        return quantized

    def _stronger(self, left: DetectedNote, right: DetectedNote) -> DetectedNote:
        if right.confidence > left.confidence:
            return right
        return left

    def _midi(self, note: DetectedNote) -> int | None:
        return PitchConverter.hz_to_midi(note.frequency_hz)
