from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class GroundTruthNote:
    sample_group: str
    sample_name: str
    prediction_json: Path
    onset_s: float
    midi_number: int
    allowed_positions: list[tuple[int, int]]


@dataclass(frozen=True)
class PredictedNote:
    onset_s: float
    midi_number: int | None
    string_number: int | None
    fret: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TaBee note-event JSON against manual ground-truth annotations."
    )
    parser.add_argument("ground_truth_csv", help="CSV with sample annotations.")
    parser.add_argument(
        "--onset-tolerance",
        type=float,
        default=0.08,
        help="Maximum onset matching distance in seconds. Default: 0.08.",
    )
    return parser.parse_args()


def read_ground_truth(path: Path) -> list[GroundTruthNote]:
    rows: list[GroundTruthNote] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "sample_group",
            "sample_name",
            "prediction_json",
            "onset_s",
            "midi_number",
            "allowed_positions",
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Ground-truth CSV is missing columns: {', '.join(sorted(missing))}")

        for row in reader:
            rows.append(
                GroundTruthNote(
                    sample_group=row["sample_group"].strip(),
                    sample_name=row["sample_name"].strip(),
                    prediction_json=Path(row["prediction_json"].strip()),
                    onset_s=float(row["onset_s"]),
                    midi_number=int(row["midi_number"]),
                    allowed_positions=parse_positions(row["allowed_positions"]),
                )
            )
    return rows


def parse_positions(value: str) -> list[tuple[int, int]]:
    positions: list[tuple[int, int]] = []
    for item in value.split("|"):
        item = item.strip()
        if not item:
            continue
        string_value, fret_value = item.split(":", 1)
        positions.append((int(string_value), int(fret_value)))
    return positions


def read_predictions(path: Path) -> list[PredictedNote]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    notes: list[PredictedNote] = []
    for event in payload.get("noteEvents", []):
        if event.get("isRest"):
            continue
        notes.append(
            PredictedNote(
                onset_s=float(event["time"]),
                midi_number=event.get("midiNumber"),
                string_number=event.get("stringNumber"),
                fret=event.get("fret"),
            )
        )
    return notes


def group_by_prediction(rows: Iterable[GroundTruthNote]) -> dict[Path, list[GroundTruthNote]]:
    grouped: dict[Path, list[GroundTruthNote]] = {}
    for row in rows:
        grouped.setdefault(row.prediction_json, []).append(row)
    return grouped


def match_notes(
    expected: list[GroundTruthNote],
    predicted: list[PredictedNote],
    tolerance: float,
) -> tuple[list[tuple[GroundTruthNote, PredictedNote]], int]:
    matches: list[tuple[GroundTruthNote, PredictedNote]] = []
    used_predictions: set[int] = set()

    for expected_note in sorted(expected, key=lambda item: item.onset_s):
        best_index: int | None = None
        best_distance = tolerance + 1.0
        for index, predicted_note in enumerate(predicted):
            if index in used_predictions:
                continue
            distance = abs(predicted_note.onset_s - expected_note.onset_s)
            if distance <= tolerance and distance < best_distance:
                best_index = index
                best_distance = distance
        if best_index is not None:
            used_predictions.add(best_index)
            matches.append((expected_note, predicted[best_index]))

    false_positives = len(predicted) - len(used_predictions)
    return matches, false_positives


def safe_percent(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator * 100.0 / denominator


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def octave_error(expected_midi: int, predicted_midi: int | None) -> bool:
    if predicted_midi is None:
        return False
    return predicted_midi != expected_midi and abs(predicted_midi - expected_midi) % 12 == 0


def main() -> int:
    args = parse_args()
    csv_path = Path(args.ground_truth_csv)
    rows = read_ground_truth(csv_path)
    root = csv_path.parent

    group_stats: dict[str, dict[str, int]] = {}

    for prediction_path, expected_rows in group_by_prediction(rows).items():
        resolved_prediction_path = prediction_path
        if not resolved_prediction_path.is_absolute():
            resolved_prediction_path = root.parent / prediction_path
        predicted_rows = read_predictions(resolved_prediction_path)
        matches, false_positives = match_notes(expected_rows, predicted_rows, args.onset_tolerance)

        sample_group = expected_rows[0].sample_group
        stats = group_stats.setdefault(
            sample_group,
            {
                "expected": 0,
                "predicted": 0,
                "matches": 0,
                "false_positives": 0,
                "pitch_correct": 0,
                "octave_errors": 0,
                "exact_assignment": 0,
                "equivalent_assignment": 0,
                "unplayable": 0,
            },
        )

        stats["expected"] += len(expected_rows)
        stats["predicted"] += len(predicted_rows)
        stats["matches"] += len(matches)
        stats["false_positives"] += false_positives

        for expected_note, predicted_note in matches:
            if predicted_note.midi_number == expected_note.midi_number:
                stats["pitch_correct"] += 1
            if octave_error(expected_note.midi_number, predicted_note.midi_number):
                stats["octave_errors"] += 1

            predicted_position = (
                None
                if predicted_note.string_number is None or predicted_note.fret is None
                else (int(predicted_note.string_number), int(predicted_note.fret))
            )
            if predicted_position is None:
                stats["unplayable"] += 1
            elif predicted_position in expected_note.allowed_positions:
                stats["equivalent_assignment"] += 1
                if expected_note.allowed_positions and predicted_position == expected_note.allowed_positions[0]:
                    stats["exact_assignment"] += 1

    print_onset_pitch_table(group_stats)
    print()
    print_fretboard_table(group_stats)
    return 0


def print_onset_pitch_table(group_stats: dict[str, dict[str, int]]) -> None:
    print("| Dataset Group | Onset Precision | Onset Recall | F1 | Pitch Accuracy | Octave Errors |")
    print("|---|---:|---:|---:|---:|---:|")
    for group_name, stats in group_stats.items():
        precision = safe_percent(stats["matches"], stats["matches"] + stats["false_positives"])
        recall = safe_percent(stats["matches"], stats["expected"])
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        pitch_accuracy = safe_percent(stats["pitch_correct"], stats["matches"])
        print(
            f"| {group_name} | {format_percent(precision)} | {format_percent(recall)} | "
            f"{format_percent(f1)} | {format_percent(pitch_accuracy)} | {stats['octave_errors']} |"
        )


def print_fretboard_table(group_stats: dict[str, dict[str, int]]) -> None:
    print("| Dataset Group | Exact Assignment Accuracy | Equivalent Assignment Accuracy | Unplayable Note Rate |")
    print("|---|---:|---:|---:|")
    for group_name, stats in group_stats.items():
        exact = safe_percent(stats["exact_assignment"], stats["matches"])
        equivalent = safe_percent(stats["equivalent_assignment"], stats["matches"])
        unplayable = safe_percent(stats["unplayable"], stats["matches"])
        print(f"| {group_name} | {format_percent(exact)} | {format_percent(equivalent)} | {format_percent(unplayable)} |")


if __name__ == "__main__":
    raise SystemExit(main())
