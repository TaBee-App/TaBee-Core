from music.PitchConverter import PitchConverter
from tabGeneration.domain.BassFretboard import BassFretboard

def run_unit_tests():
    print("--- 1. TEST: hz_to_midi ---")
    # Standart akort frekansları (A4 = 440Hz -> MIDI 69, E1 = 41.2Hz -> MIDI 28)
    print(f"440.0 Hz -> MIDI: {PitchConverter.hz_to_midi(440.0)} (Beklenen: 69)")
    print(f"41.2 Hz  -> MIDI: {PitchConverter.hz_to_midi(41.2)} (Beklenen: 28)")
    print(f"0 Hz     -> MIDI: {PitchConverter.hz_to_midi(0)} (Beklenen: -1)")

    print("\n--- 2. TEST: hz_to_note_name ---")
    print(f"440.0 Hz -> Nota: {PitchConverter.hz_to_note_name(440.0)} (Beklenen: A4)")
    print(f"41.2 Hz  -> Nota: {PitchConverter.hz_to_note_name(41.2)} (Beklenen: E1)")

    print("\n--- 3. TEST: midi_to_note_name ---")
    print(f"MIDI 69 -> Nota: {PitchConverter.midi_to_note_name(69)} (Beklenen: A4)")
    print(f"MIDI 28 -> Nota: {PitchConverter.midi_to_note_name(28)} (Beklenen: E1)")

    print("\n--- 4. TEST: get_candidates (BassFretboard) ---")
    fretboard = BassFretboard()
    
    # MIDI 28 (E1): Beklenti -> Sadece 4. tel (E), 0. perde
    print(f"MIDI 28 (E1) Adayları: {fretboard.get_candidates(28)}")
    
    # MIDI 33 (A1): Beklenti -> 4. tel 5. perde VE 3. tel (A) 0. perde
    print(f"MIDI 33 (A1) Adayları: {fretboard.get_candidates(33)}")
    
    # MIDI -1 (Es/Susturma): Beklenti -> Rest dönmeli
    print(f"MIDI -1 (Es) Adayları: {fretboard.get_candidates(-1)}")

if __name__ == "__main__":
    run_unit_tests()