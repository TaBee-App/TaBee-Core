import librosa
import numpy as np
from ..ports.AudioReader import AudioReader


class LibrosaAudioReader(AudioReader):

    def load_mono(self, path: str, sample_rate: int = 44100):
        y, sr = librosa.load(path, sr=sample_rate, mono=True)
        return y.astype(np.float32), sr
