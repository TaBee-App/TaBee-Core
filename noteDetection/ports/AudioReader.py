from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class AudioReader(ABC):

    @abstractmethod
    def load_mono(
        self,
        path: str,
        sample_rate: int = 44100
    ) -> Tuple[np.ndarray, int]:
        pass
