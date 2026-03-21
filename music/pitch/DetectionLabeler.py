from typing import List

from noteDetection.domain.DetectionResult import DetectionResult
from music.domain.DetectedPitch import DetectedPitch
from music.pitch.PitchConverter import PitchConverter


class DetectionLabeler:
    """
    Converts raw detected Hz-based notes into named musical notes.

    This keeps:
    - noteDetection layer raw and DSP-oriented
    - music layer responsible for interpretation
    """

    def label(self, result: DetectionResult) -> List[DetectedPitch]:
        labeled: List[DetectedPitch] = []

        for detected_note in result.notes:
            musical_note = PitchConverter.hz_to_musical_note(
                detected_note.frequency_hz
            )

            labeled.append(
                DetectedPitch(
                    detected=detected_note,
                    musical=musical_note,
                )
            )

        return labeled