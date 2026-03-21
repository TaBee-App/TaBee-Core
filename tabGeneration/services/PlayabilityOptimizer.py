class PlayabilityOptimizer:
    def __init__(self):
        # Maliyet ağırlıkları (Projenin ileriki aşamalarında bu katsayılarla oynayarak optimizasyonu hassaslaştırabiliriz)
        self.weight_fret = 1.0    # Perdeler arası el kaydırma maliyeti
        self.weight_string = 1.5  # Teller arası parmak atlama maliyeti

    def calculate_cost(self, prev_pos: dict, curr_pos: dict) -> float:
        """
        Bir önceki (Tel, Perde) konumundan, yeni konuma geçişin zorluğunu hesaplar.
        """
        # Eğer Rest (Es) notasıysa maliyet sıfırdır, el serbesttir.
        if prev_pos.get('string') == 'Rest' or curr_pos.get('string') == 'Rest':
            return 0.0

        # Boş teller (Fret 0) geçişleri kolaylaştırır çünkü eli klavyede kaydırmak için zaman kazandırır.
        if prev_pos['fret'] == 0 or curr_pos['fret'] == 0:
            fret_diff = 0
        else:
            fret_diff = abs(prev_pos['fret'] - curr_pos['fret'])
        
        string_diff = abs(prev_pos['string'] - curr_pos['string'])
        
        return (fret_diff * self.weight_fret) + (string_diff * self.weight_string)

    def optimize(self, candidates_sequence: list) -> list:
        """
        Viterbi algoritması kullanarak şarkı başından sonuna kadar
        en düşük maliyetli (en kolay çalınabilir) rotayı bulur.
        """
        if not candidates_sequence:
            return []

        # DP (Dinamik Programlama) Tablosu
        # dp[nota_index][aday_index] = (O_ana_kadarki_toplam_maliyet, Bir_onceki_en_iyi_adayin_indexi)
        dp = [{i: (0, None) for i in range(len(candidates_sequence[0]))}]

        # 1. İleriye Doğru Tarama (Tüm rotaların maliyetini hesapla)
        for i in range(1, len(candidates_sequence)):
            current_candidates = candidates_sequence[i]
            prev_candidates = candidates_sequence[i-1]
            current_dp = {}
            
            for curr_idx, curr_cand in enumerate(current_candidates):
                min_cost = float('inf')
                best_prev_idx = None
                
                for prev_idx, prev_cand in enumerate(prev_candidates):
                    prev_cost = dp[i-1][prev_idx][0]
                    transition_cost = self.calculate_cost(prev_cand, curr_cand)
                    total_cost = prev_cost + transition_cost
                    
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev_idx = prev_idx
                        
                current_dp[curr_idx] = (min_cost, best_prev_idx)
            dp.append(current_dp)

        # 2. Geriye Doğru Takip (En ucuz rotayı sondan başa doğru topla)
        optimal_path = []
        last_dp = dp[-1]
        
        # Son notadaki en düşük maliyetli adayı bul
        curr_idx = min(last_dp.keys(), key=lambda k: last_dp[k][0])
        
        # Zinciri geriye doğru sararak en iyi yolu çıkart
        for i in range(len(candidates_sequence) - 1, -1, -1):
            optimal_path.append(candidates_sequence[i][curr_idx])
            curr_idx = dp[i][curr_idx][1] # Bir önceki adımdaki en iyi adaya geç
            
        # Listeyi baştan sona doğru olacak şekilde tersine çevir
        optimal_path.reverse()
        return optimal_path