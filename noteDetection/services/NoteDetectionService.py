from ..ports.AudioReader import AudioReader
from ..services.NoteDetectionPipeline import NoteDetectionPipeline
from ..domain.DetectionResult import DetectionResult


class NoteDetectionService:

    def __init__(self, audio_reader: AudioReader, pipeline: NoteDetectionPipeline):
        self._audio_reader = audio_reader
        self._pipeline = pipeline

    def analyze_file(self, path: str) -> DetectionResult:
        y, sr = self._audio_reader.load_mono(path)
        return self._pipeline.detect(y, sr)