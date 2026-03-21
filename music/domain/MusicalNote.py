from __future__ import annotations


class MusicalNote:
    """
    Immutable value object representing a musical pitch.

    Stored as:
    - midi_note: integer MIDI note number
    - name: pitch-class name (sharp naming)
    - octave: scientific pitch octave
    - frequency_hz: original observed frequency
    """

    def __init__(self, midi_note: int, name: str, octave: int, frequency_hz: float):
        self._midi_note = int(midi_note)
        self._name = str(name)
        self._octave = int(octave)
        self._frequency_hz = float(frequency_hz)

    @property
    def midi_note(self) -> int:
        return self._midi_note

    @property
    def name(self) -> str:
        return self._name

    @property
    def octave(self) -> int:
        return self._octave

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def full_name(self) -> str:
        return f"{self._name}{self._octave}"

    def __repr__(self) -> str:
        return (
            f"MusicalNote("
            f"midi_note={self._midi_note}, "
            f"name='{self._name}', "
            f"octave={self._octave}, "
            f"frequency_hz={self._frequency_hz:.2f}"
            f")"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MusicalNote):
            return False

        return (
            self._midi_note == other._midi_note
            and self._name == other._name
            and self._octave == other._octave
            and abs(self._frequency_hz - other._frequency_hz) < 1e-9
        )

    def __hash__(self) -> int:
        return hash((self._midi_note, self._name, self._octave))