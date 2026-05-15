import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noteDetection import LibrosaAudioReader, NoteDetectionPipeline, NoteDetectionService
from tabGeneration import TabGenerationService, TabRenderer


if __name__ == "__main__":
    service = NoteDetectionService(
        audio_reader=LibrosaAudioReader(),
        pipeline=NoteDetectionPipeline(),
    )
    tab_service = TabGenerationService()
    renderer = TabRenderer()

    result = service.analyze_file("queen.wav")
    assignments = tab_service.generate_from_detection_result(result)

    print("Tempo:", result.tempo_bpm)
    print("Detected notes:", len(result.notes))
    print()
    print("time(s)   note   midi   string   fret   conf")

    for assignment in assignments[:64]:
        string_label = "Rest" if assignment.position.is_rest else assignment.position.string_number
        note_label = assignment.note_name or "Rest"
        midi_label = assignment.midi_note if assignment.midi_note is not None else "Rest"
        print(
            f"{assignment.time_s:7.3f}   "
            f"{note_label:4s}   "
            f"{str(midi_label):>4s}   "
            f"{str(string_label):>6s}   "
            f"{assignment.position.fret:4d}   "
            f"{assignment.confidence:4.2f}"
        )

    print()
    print(renderer.render_ascii(assignments[:64]))
