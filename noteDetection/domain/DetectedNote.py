class DetectedNote:
    def __init__(self, time_s: float, frequency_hz: float, confidence: float):
        self._time_s = float(time_s)
        self._frequency_hz = float(frequency_hz)
        self._confidence = float(confidence)

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def confidence(self) -> float:
        return self._confidence