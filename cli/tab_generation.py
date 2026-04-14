from noteDetection import LibrosaAudioReader, NoteDetectionPipeline, NoteDetectionService
from tabGeneration import TabGenerationService


if __name__ == "__main__":
    service = NoteDetectionService(
        audio_reader=LibrosaAudioReader(),
        pipeline=NoteDetectionPipeline(),
    )
    tab_service = TabGenerationService()

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
