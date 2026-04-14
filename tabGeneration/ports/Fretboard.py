from abc import ABC, abstractmethod
from typing import Optional

from ..domain.FretPosition import FretPosition


class Fretboard(ABC):

    @abstractmethod
    def get_candidates(self, target_midi: Optional[int]) -> list[FretPosition]:
        pass
