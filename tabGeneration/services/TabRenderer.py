from __future__ import annotations

from typing import Sequence

from tabGeneration.domain.TabNoteAssignment import TabNoteAssignment


class TabRenderer:
    """Renders optimized bass tab assignments as a compact terminal tablature."""

    _STRING_LABELS = {
        1: "G",
        2: "D",
        3: "A",
        4: "E",
        5: "B",
    }

    def render_ascii(
        self,
        assignments: Sequence[TabNoteAssignment],
        *,
        notes_per_line: int = 16,
        string_numbers: Sequence[int] = (1, 2, 3, 4),
        string_labels: dict[int, str] | None = None,
    ) -> str:
        if not assignments:
            return "No playable notes detected."

        labels = string_labels or self._STRING_LABELS
        lines: list[str] = []
        for start in range(0, len(assignments), notes_per_line):
            chunk = assignments[start:start + notes_per_line]
            lines.append(self._render_chunk(chunk, string_numbers=string_numbers, string_labels=labels))
        return "\n\n".join(lines)

    def _render_chunk(
        self,
        assignments: Sequence[TabNoteAssignment],
        *,
        string_numbers: Sequence[int],
        string_labels: dict[int, str],
    ) -> str:
        width = max(3, max(len(str(a.position.fret)) for a in assignments) + 1)
        rows = {string_number: [] for string_number in string_numbers}
        time_cells: list[str] = []
        note_cells: list[str] = []

        for assignment in assignments:
            time_cells.append(f"{assignment.time_s:.2f}".rjust(width))
            note_cells.append((assignment.note_name or "-").rjust(width))
            for string_number in string_numbers:
                if assignment.position.string_number == string_number:
                    rows[string_number].append(str(assignment.position.fret).rjust(width, "-"))
                else:
                    rows[string_number].append("-" * width)

        rendered = [
            "time " + " ".join(time_cells),
            "note " + " ".join(note_cells),
        ]
        for string_number in string_numbers:
            rendered.append(f"{string_labels[string_number]}|   " + " ".join(rows[string_number]))
        return "\n".join(rendered)
