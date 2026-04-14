from __future__ import annotations

from typing import Sequence

from ..domain.FretPosition import FretPosition
from ..ports.TabOptimizer import TabOptimizer


class PlayabilityOptimizer(TabOptimizer):
    def __init__(self, weight_fret: float = 1.0, weight_string: float = 1.5):
        self._weight_fret = float(weight_fret)
        self._weight_string = float(weight_string)

    def calculate_cost(
        self,
        previous_position: FretPosition,
        current_position: FretPosition,
    ) -> float:
        """
        Estimate physical movement cost between two fretboard positions.

        Open strings reduce hand-shift cost because the fretting hand can reset.
        Rests are treated as free transitions.
        """
        if previous_position.is_rest or current_position.is_rest:
            return 0.0

        if previous_position.fret == 0 or current_position.fret == 0:
            fret_distance = 0
        else:
            fret_distance = abs(previous_position.fret - current_position.fret)

        string_distance = abs(
            int(previous_position.string_number) - int(current_position.string_number)
        )

        return (
            (fret_distance * self._weight_fret)
            + (string_distance * self._weight_string)
        )

    def optimize(
        self,
        candidate_sequence: Sequence[Sequence[FretPosition]],
    ) -> list[FretPosition]:
        if not candidate_sequence:
            return []

        dp: list[dict[int, tuple[float, int | None]]] = [
            {index: (0.0, None) for index in range(len(candidate_sequence[0]))}
        ]

        for note_index in range(1, len(candidate_sequence)):
            current_candidates = candidate_sequence[note_index]
            previous_candidates = candidate_sequence[note_index - 1]
            current_dp: dict[int, tuple[float, int | None]] = {}

            for current_index, current_candidate in enumerate(current_candidates):
                min_cost = float("inf")
                best_previous_index: int | None = None

                for previous_index, previous_candidate in enumerate(previous_candidates):
                    previous_cost = dp[note_index - 1][previous_index][0]
                    transition_cost = self.calculate_cost(
                        previous_candidate,
                        current_candidate,
                    )
                    total_cost = previous_cost + transition_cost

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_previous_index = previous_index

                current_dp[current_index] = (min_cost, best_previous_index)

            dp.append(current_dp)

        optimal_path: list[FretPosition] = []
        current_index = min(dp[-1].keys(), key=lambda idx: dp[-1][idx][0])

        for note_index in range(len(candidate_sequence) - 1, -1, -1):
            optimal_path.append(candidate_sequence[note_index][current_index])
            previous_index = dp[note_index][current_index][1]
            if previous_index is None:
                break
            current_index = previous_index

        optimal_path.reverse()
        return optimal_path
