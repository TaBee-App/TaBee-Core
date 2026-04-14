from noteDetection import (
    NoteDetectionService,
    NoteDetectionPipeline,
    LibrosaAudioReader,
)


if __name__ == "__main__":
    service = NoteDetectionService(
        audio_reader=LibrosaAudioReader(),
        pipeline=NoteDetectionPipeline(),
    )

    result = service.analyze_file("test2.wav")

    print("Tempo:", result.tempo_bpm)
    print("Onsets:", len(result.onset_times_s))
    print("Notes:", len(result.notes))

    for note in result.notes[:32]:
        print(
            f"time={note.time_s:.3f}s "
            f"freq={note.frequency_hz:.1f}Hz "
            f"conf={note.confidence:.2f}"
        )