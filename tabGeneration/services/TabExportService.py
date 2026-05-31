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
        beat_seconds = 60.0 / float(detection.tempo_bpm) if detection.tempo_bpm else 0.5
        min_rest_seconds = max(0.06, beat_seconds / 4.0)

        if assignments and assignments[0].time_s >= min_rest_seconds:
            note_events.append(
                {
                    "isRest": True,
                    "time": 0.0,
                    "duration": round(assignments[0].time_s, 4),
                    "frequency": None,
                    "confidence": None,
                    "noteName": None,
                    "midiNumber": None,
                    "fret": None,
                    "stringNumber": None,
                }
            )

        for index, assignment in enumerate(assignments):
            next_time = (
                assignments[index + 1].time_s
                if index + 1 < len(assignments)
                else None
            )
            gap_duration = None if next_time is None else max(0.0, next_time - assignment.time_s)
            duration = assignment.duration_s if assignment.duration_s is not None else gap_duration
            if gap_duration is not None and duration is not None:
                duration = min(duration, gap_duration)

            note_events.append(
                {
                    "isRest": False,
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

            if next_time is not None and duration is not None:
                rest_start = assignment.time_s + duration
                rest_duration = max(0.0, next_time - rest_start)
                if rest_duration >= min_rest_seconds:
                    note_events.append(
                        {
                            "isRest": True,
                            "time": round(rest_start, 4),
                            "duration": round(rest_duration, 4),
                            "frequency": None,
                            "confidence": None,
                            "noteName": None,
                            "midiNumber": None,
                            "fret": None,
                            "stringNumber": None,
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
