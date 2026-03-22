class PluckCandidate:
    def __init__(
        self,
        time_s: float,
        strength: float,
        envelope_strength: float,
        spectral_strength: float,
    ):
        self._time_s = float(time_s)
        self._strength = float(strength)
        self._envelope_strength = float(envelope_strength)
        self._spectral_strength = float(spectral_strength)

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def strength(self) -> float:
        return self._strength

    @property
    def envelope_strength(self) -> float:
        return self._envelope_strength

    @property
    def spectral_strength(self) -> float:
        return self._spectral_strength
