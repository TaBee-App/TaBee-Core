from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from ..domain.PluckCandidate import PluckCandidate


class PitchEstimator(ABC):

    @abstractmethod
    def estimate(
        self, y: np.ndarray, sr: int, candidates: List[PluckCandidate]
    ) -> Tuple[List[float], List[float]]:
        pass
