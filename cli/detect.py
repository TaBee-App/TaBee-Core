from noteDetection import (
    NoteDetectionService,
    NoteDetectionPipeline,
    LibrosaAudioReader,
)

from music.PitchConverter import PitchConverter
from tabGeneration.domain.BassFretboard import BassFretboard
from tabGeneration.services.PlayabilityOptimizer import PlayabilityOptimizer

# --- OKTAV FİLTRESİ ---
def fix_bass_octave(midi_val: int, threshold: int = 45) -> int:
    """
    Nota belirlenen eşik değerinden (MIDI 45) yüksekse, 
    bas gitar aralığına inene kadar 1 oktav (12 birim) düşürür.
    """
    if midi_val <= 0:
        return midi_val
        
    corrected_midi = midi_val
    while corrected_midi > threshold:
        corrected_midi -= 12
    return corrected_midi

if __name__ == "__main__":
    service = NoteDetectionService(
        audio_reader=LibrosaAudioReader(),
        pipeline=NoteDetectionPipeline(),
    )

    fretboard = BassFretboard()
    optimizer = PlayabilityOptimizer()

    result = service.analyze_file("cli/queen-another-one-bites-the-dust-bass-only.wav")

    print("Tempo:", result.tempo_bpm)
    print("Onsets:", len(result.onset_times_s))

    # Tüm şarkının aday listesini tutacağımız büyük dizi
    all_candidates_sequence = []
    
    # 1. Aşama: Tüm notalar için adayları (tel, perde kombinasyonları) bul
    for note in result.notes:
        raw_midi = PitchConverter.hz_to_midi(note.frequency_hz)
        corrected_midi = fix_bass_octave(raw_midi)
        candidates = fretboard.get_candidates(corrected_midi)
        
        # Eğer ses tanıma hatasından dolayı aday bulunamazsa sistemi çökertmemek için "Rest" (Es) ekle
        if not candidates:
            candidates = [{"string": "Rest", "fret": 0}]
            
        all_candidates_sequence.append(candidates)

    # 2. Aşama: Viterbi Algoritması ile en iyi (en az yoran) rotayı çiz
    optimal_tab_sequence = optimizer.optimize(all_candidates_sequence)

    # 3. Aşama: İlk 10 notanın sonucunu ekrana yazdır
    print("\n--- İLK 10 NOTA İÇİN VITERBI İLE OPTİMİZE EDİLMİŞ TABLATÜR ÇIKTISI ---")
    for i in range(len(result.notes)):
    # for i in range(10):
        note = result.notes[i]
        corrected_midi = fix_bass_octave(PitchConverter.hz_to_midi(note.frequency_hz))
        note_name = PitchConverter.midi_to_note_name(corrected_midi)
        chosen_pos = optimal_tab_sequence[i]
        
        print(
            f"Zaman: {note.time_s:.3f}s | "
            f"Nota: {note_name:<3} (MIDI: {corrected_midi}) | "
            f"SEÇİLEN -> Tel: {chosen_pos['string']}, Perde: {chosen_pos['fret']}"
        )