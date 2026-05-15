from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from noteDetection.domain.DetectionResult import DetectionResult
from tabGeneration.domain.TabNoteAssignment import TabNoteAssignment


class TabExportService:
    """
    Builds the JSON structure used by the backend tables:
    tab_data: tuning, estimated_tempo, json_data
    note_events: time, frequency, confidence, note_name, midi_number, fret, string_number
    """

    def to_json_dict(
        self,
        *,
        source_audio: str,
        detection: DetectionResult,
        assignments: Sequence[TabNoteAssignment],
        tuning: str = "EADG",
    ) -> dict[str, Any]:
        note_events = []
        for index, assignment in enumerate(assignments):
            next_time = (
                assignments[index + 1].time_s
                if index + 1 < len(assignments)
                else None
            )
            duration = None if next_time is None else max(0.0, next_time - assignment.time_s)
            note_events.append(
                {
                    "time": round(assignment.time_s, 4),
                    "duration": None if duration is None else round(duration, 4),
                    "frequency": round(assignment.frequency_hz, 3),
                    "confidence": round(assignment.confidence, 4),
                    "noteName": assignment.note_name,
                    "midiNumber": assignment.midi_note,
                    "fret": None if assignment.position.is_rest else assignment.position.fret,
                    "stringNumber": assignment.position.string_number,
                }
            )

        return {
            "sourceAudio": source_audio,
            "instrument": "bass",
            "tuning": tuning,
            "estimatedTempo": detection.tempo_bpm,
            "algorithm": {
                "onsetDetection": "bass envelope rise + spectral flux",
                "pitchDetection": "librosa.pyin median pitch per onset window",
                "tabOptimization": "Viterbi dynamic programming with playability transition costs",
            },
            "summary": {
                "detectedOnsets": len(detection.onset_times_s),
                "detectedNotes": len(assignments),
                "playableNotes": sum(1 for item in assignments if not item.position.is_rest),
            },
            "noteEvents": note_events,
        }
