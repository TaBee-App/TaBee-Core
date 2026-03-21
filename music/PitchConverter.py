import librosa
import numpy as np

class PitchConverter:
    @staticmethod
    def hz_to_midi(freq_hz: float) -> int:
        """
        Frekans (Hz) değerini en yakın MIDI nota numarasına çevirir.
        Geçersiz veya 0 frekans gelirse -1 döndürür.
        """
        if freq_hz <= 0:
            return -1
        
        # librosa'nın hz_to_midi fonksiyonu küsuratlı değer dönebilir,
        # biz bunu en yakın tam sayı MIDI notasına yuvarlıyoruz.
        midi_float = librosa.hz_to_midi(freq_hz)
        return int(np.round(midi_float))

    @staticmethod
    def hz_to_note_name(freq_hz: float) -> str:
        """
        Frekans değerini nota ismine çevirir (Örn: 'E1', 'A2').
        """
        if freq_hz <= 0:
            return "Rest"
        
        return librosa.hz_to_note(freq_hz)

    @staticmethod
    def midi_to_note_name(midi_val: int) -> str:
        """
        MIDI numarasını nota ismine çevirir (Örn: 38 -> 'D2').
        """
        if midi_val <= 0:
            return "Rest"
        
        return librosa.midi_to_note(midi_val)