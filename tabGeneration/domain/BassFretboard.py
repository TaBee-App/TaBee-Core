class BassFretboard:
    def __init__(self, max_frets: int = 24):
        # 4 Telli Standart Bas Gitar Akordu (Açık tel MIDI numaraları)
        # 4. Tel (En kalın): E1 (MIDI 28)
        # 3. Tel: A1 (MIDI 33)
        # 2. Tel: D2 (MIDI 38)
        # 1. Tel (En ince): G2 (MIDI 43)
        self.string_tunings = {
            1: 43, # G
            2: 38, # D
            3: 33, # A
            4: 28  # E
        }
        self.max_frets = max_frets

    def get_candidates(self, target_midi: int) -> list:
        """
        Verilen bir MIDI notasının bas gitar üzerinde çalınabileceği 
        tüm (tel, perde) kombinasyonlarını bulur.
        """
        candidates = []
        
        # Es (Rest) notası geldiyse
        if target_midi == -1:
            return [{"string": "Rest", "fret": 0}]

        # Tüm telleri tek tek kontrol et
        for string_num, base_midi in self.string_tunings.items():
            fret = target_midi - base_midi
            # Eğer nota bu telde çalınabiliyorsa (perde 0 ile max_frets arasındaysa)
            if 0 <= fret <= self.max_frets:
                candidates.append({
                    "string": string_num,
                    "fret": fret
                })
                
        return candidates