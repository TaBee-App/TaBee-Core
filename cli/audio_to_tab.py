from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from noteDetection import LibrosaAudioReader, NoteDetectionPipeline, NoteDetectionService
from tabGeneration import TabGenerationService, TabRenderer
from tabGeneration.services.TabExportService import TabExportService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a monophonic bass WAV/audio file into JSON note events and an ASCII tablature preview."
    )
    parser.add_argument("audio_path", help="Path to input audio file, for example cli/test.wav")
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional output JSON path. Defaults to <audio filename>.tab.json next to the input file.",
    )
    parser.add_argument(
        "--ascii-out",
        default=None,
        help="Optional output text path for terminal tablature. Defaults to <audio filename>.tab.txt.",
    )
    parser.add_argument(
        "--tuning",
        default="EADG",
        choices=["EADG", "BEADG", "CGCF", "EBABDBGB"],
        help="Bass tuning. EADG is standard 4-string bass, BEADG is 5-string bass, CGCF is dropped C, EBABDBGB is half-step down.",
    )
    parser.add_argument("--notes-per-line", type=int, default=16, help="ASCII tab notes per rendered line.")
    parser.add_argument("--print-json", action="store_true", help="Print JSON output to terminal too.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    audio_path = Path(args.audio_path)
    if not audio_path.exists():
        raise SystemExit(f"Audio file not found: {audio_path}")

    detection_service = NoteDetectionService(
        audio_reader=LibrosaAudioReader(),
        pipeline=NoteDetectionPipeline(),
    )
    tab_service = TabGenerationService(tuning=args.tuning)
    renderer = TabRenderer()
    exporter = TabExportService()

    detection = detection_service.analyze_file(str(audio_path))
    assignments = tab_service.generate_from_detection_result(detection)

    json_data = exporter.to_json_dict(
        source_audio=str(audio_path),
        detection=detection,
        assignments=assignments,
        tuning=args.tuning,
    )
    string_numbers = (1, 2, 3, 4, 5) if args.tuning == "BEADG" else (1, 2, 3, 4)
    string_labels = {1: "Gb", 2: "Db", 3: "Ab", 4: "Eb"} if args.tuning == "EBABDBGB" else None
    ascii_tab = renderer.render_ascii(
        assignments,
        notes_per_line=args.notes_per_line,
        string_numbers=string_numbers,
        string_labels=string_labels,
    )

    json_out = Path(args.json_out) if args.json_out else audio_path.with_suffix(".tab.json")
    ascii_out = Path(args.ascii_out) if args.ascii_out else audio_path.with_suffix(".tab.txt")

    json_out.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    ascii_out.write_text(ascii_tab, encoding="utf-8")

    print(f"Tempo: {detection.tempo_bpm} BPM")
    print(f"Detected notes: {len(assignments)}")
    print(f"Playable notes: {json_data['summary']['playableNotes']}")
    print(f"JSON output: {json_out}")
    print(f"ASCII tab output: {ascii_out}")
    print()
    print(ascii_tab)

    if args.print_json:
        print()
        print(json.dumps(json_data, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
