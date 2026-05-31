import unittest

from noteDetection.domain.DetectedNote import DetectedNote
from noteDetection.domain.DetectionResult import DetectionResult
from noteDetection.services.NoteEventPostProcessor import (
    NoteEventPostProcessor,
    NotePostProcessConfig,
)


class NoteEventPostProcessorTest(unittest.TestCase):
    def test_collapses_duplicate_events_on_same_grid_slot(self):
        result = DetectionResult(
            tempo_bpm=120,
            onset_times_s=[1.001, 1.018, 1.51],
            pitch_hz=[45.0, 46.0, 55.0],
            notes=[
                DetectedNote(1.001, 45.0, 0.4, 0.08),
                DetectedNote(1.018, 46.0, 0.9, 0.08),
                DetectedNote(1.51, 55.0, 0.8, 0.08),
            ],
        )

        processor = NoteEventPostProcessor(NotePostProcessConfig())
        processed = processor.process(result)

        self.assertEqual(2, len(processed.notes))
        self.assertEqual(46.0, processed.notes[0].frequency_hz)
        self.assertAlmostEqual(1.0, processed.notes[0].time_s)
        self.assertAlmostEqual(1.5, processed.notes[1].time_s)

    def test_filters_unpitched_and_low_confidence_events(self):
        result = DetectionResult(
            tempo_bpm=100,
            onset_times_s=[0.0, 0.3, 0.6],
            pitch_hz=[0.0, 45.0, 55.0],
            notes=[
                DetectedNote(0.0, 0.0, 0.9, 0.1),
                DetectedNote(0.3, 45.0, 0.01, 0.1),
                DetectedNote(0.6, 55.0, 0.5, 0.1),
            ],
        )

        processor = NoteEventPostProcessor(NotePostProcessConfig(min_confidence=0.02))
        processed = processor.process(result)

        self.assertEqual(1, len(processed.notes))
        self.assertEqual(55.0, processed.notes[0].frequency_hz)


if __name__ == "__main__":
    unittest.main()
