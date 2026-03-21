from __future__ import annotations

import math
from typing import Optional

from music.domain.MusicalNote import MusicalNote


class PitchConverter:
    """
    Pure pitch / note conversion logic.

    Responsibilities:
    - Hz -> MIDI
    - MIDI -> Hz
    - MIDI -> note name + octave
    - Hz -> MusicalNote
    - Hz -> nearest equal-tempered snapped frequency
    """

    _NOTE_NAMES_SHARP = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    @staticmethod
    def hz_to_midi(frequency_hz: float) -> Optional[int]:
        if frequency_hz is None:
            return None

        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            return None

        midi = 69.0 + 12.0 * math.log2(frequency_hz / 440.0)
        return int(round(midi))

    @staticmethod
    def midi_to_hz(midi_note: int) -> float:
        midi_note = int(midi_note)
        return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))

    @staticmethod
    def midi_to_name_octave(midi_note: int) -> tuple[str, int]:
        midi_note = int(midi_note)
        name = PitchConverter._NOTE_NAMES_SHARP[midi_note % 12]
        octave = (midi_note // 12) - 1
        return name, octave

    @staticmethod
    def hz_to_musical_note(frequency_hz: float) -> Optional[MusicalNote]:
        midi_note = PitchConverter.hz_to_midi(frequency_hz)
        if midi_note is None:
            return None

        name, octave = PitchConverter.midi_to_name_octave(midi_note)

        return MusicalNote(
            midi_note=midi_note,
            name=name,
            octave=octave,
            frequency_hz=float(frequency_hz),
        )

    @staticmethod
    def snap_hz_to_equal_temperament(frequency_hz: float) -> Optional[float]:
        midi_note = PitchConverter.hz_to_midi(frequency_hz)
        if midi_note is None:
            return None
        return PitchConverter.midi_to_hz(midi_note)