class DetectedNote:
    def __init__(
        self,
        time_s: float,
        frequency_hz: float,
        confidence: float,
        duration_s: float | None = None,
    ):
        self._time_s = float(time_s)
        self._frequency_hz = float(frequency_hz)
        self._confidence = float(confidence)
        self._duration_s = None if duration_s is None else max(0.0, float(duration_s))

    @property
    def time_s(self) -> float:
        return self._time_s

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def duration_s(self) -> float | None:
        return self._duration_s
