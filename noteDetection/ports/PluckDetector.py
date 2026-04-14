from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..domain.PluckCandidate import PluckCandidate


class PluckDetector(ABC):

    @abstractmethod
    def detect(self, y: np.ndarray, sr: int) -> List[PluckCandidate]:
        pass
