from noteDetection import (
    NoteDetectionService,
    NoteDetectionPipeline,
    LibrosaAudioReader,
)
from music.pitch.DetectionLabeler import DetectionLabeler


if __name__ == "__main__":
    service = NoteDetectionService(
        audio_reader=LibrosaAudioReader(),
        pipeline=NoteDetectionPipeline(),
    )

    result = service.analyze_file("queen.wav")

    labeled_notes = DetectionLabeler().label(result)

    for item in labeled_notes[:64]:
        detected = item.detected
        musical = item.musical

        if musical is None:
            note_name = "None"
            midi_note = "None"
        else:
            note_name = musical.full_name
            midi_note = musical.midi_note

        print(
            f"time={detected.time_s:.3f}s "
            f"freq={detected.frequency_hz:7.1f}Hz "
            f"note={note_name:4s} "
            f"midi={midi_note} "
            f"conf={detected.confidence:.2f}"
        )