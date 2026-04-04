from abc import ABC, abstractmethod
from typing import Sequence

from ..domain.FretPosition import FretPosition


class TabOptimizer(ABC):

    @abstractmethod
    def optimize(
        self,
        candidate_sequence: Sequence[Sequence[FretPosition]],
    ) -> list[FretPosition]:
        pass
