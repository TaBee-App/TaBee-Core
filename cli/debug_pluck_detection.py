from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from music.pitch.DetectionLabeler import DetectionLabeler
from noteDetection import LibrosaAudioReader, NoteDetectionPipeline, NoteDetectionService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug bass pluck detection using gap-based phrase grouping."
    )
    parser.add_argument(
        "audio",
        nargs="?",
        default="queen.wav",
        help="Audio file to analyze. Defaults to queen.wav in the cli directory.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=160,
        help="Maximum number of events to print.",
    )
    parser.add_argument(
        "--group-gap-ms",
        type=float,
        default=380.0,
        help="Start a new group when the gap from the previous event exceeds this many milliseconds.",
    )
    parser.add_argument(
        "--collapse-gap-ms",
        type=float,
        default=135.0,
        help="Merge near-duplicate adjacent events within this many milliseconds when they have the same note name.",
    )
    parser.add_argument(
        "--keep-none",
        action="store_true",
        help="Keep unpitched events instead of dropping them before grouping.",
    )
    return parser


def resolve_audio_path(audio_arg: str) -> Path:
    path = Path(audio_arg)
    if path.is_absolute():
        return path
    cli_dir = Path(__file__).resolve().parent
    return (cli_dir / path).resolve()


def compact_note_name(note_name: str) -> str:
    return "--" if note_name == "None" else note_name


def choose_better_event(left: dict, right: dict) -> dict:
    return max(
        (left, right),
        key=lambda event: (
            event["note_name"] != "--",
            event["confidence"],
            event["strength"],
        ),
    )


def collapse_events(events: list[dict], collapse_gap_s: float) -> list[dict]:
    if not events:
        return []

    collapsed: list[dict] = [dict(events[0])]
    for event in events[1:]:
        previous = collapsed[-1]
        same_note = event["note_name"] == previous["note_name"]
        close_in_time = (event["time_s"] - previous["time_s"]) <= collapse_gap_s
        if same_note and close_in_time:
            winner = choose_better_event(previous, event)
            merged = dict(winner)
            merged["merged_count"] = previous["merged_count"] + event["merged_count"]
            merged["merged_from"] = previous["merged_from"] + event["merged_from"]
            collapsed[-1] = merged
            continue

        collapsed.append(dict(event))

    return collapsed


def group_events(events: list[dict], group_gap_s: float) -> list[list[dict]]:
    if not events:
        return []

    groups: list[list[dict]] = [[events[0]]]
    for event in events[1:]:
        previous = groups[-1][-1]
        if (event["time_s"] - previous["time_s"]) > group_gap_s:
            groups.append([event])
        else:
            groups[-1].append(event)
    return groups


if __name__ == "__main__":
    args = build_parser().parse_args()
    audio_path = resolve_audio_path(args.audio)

    service = NoteDetectionService(
        audio_reader=LibrosaAudioReader(),
        pipeline=NoteDetectionPipeline(),
    )

    result = service.analyze_file(str(audio_path))
    labeled_notes = DetectionLabeler().label(result)
    onset_times = np.asarray(result.onset_times_s, dtype=float)
    iois = np.diff(onset_times) if onset_times.size >= 2 else np.asarray([])

    raw_events: list[dict] = []
    for idx, (candidate, item) in enumerate(zip(result.pluck_candidates, labeled_notes)):
        musical = item.musical
        note_name = compact_note_name(musical.full_name if musical is not None else "None")
        midi_note = str(musical.midi_note) if musical is not None else "None"

        event = {
            "idx": idx,
            "time_s": candidate.time_s,
            "strength": candidate.strength,
            "env": candidate.envelope_strength,
            "spec": candidate.spectral_strength,
            "frequency_hz": item.detected.frequency_hz,
            "confidence": item.detected.confidence,
            "note_name": note_name,
            "midi_note": midi_note,
            "merged_count": 1,
            "merged_from": [idx],
        }
        raw_events.append(event)

    filtered_events = (
        raw_events if args.keep_none else [event for event in raw_events if event["note_name"] != "--"]
    )
    collapsed_events = collapse_events(
        filtered_events, collapse_gap_s=args.collapse_gap_ms / 1000.0
    )
    groups = group_events(collapsed_events, group_gap_s=args.group_gap_ms / 1000.0)

    print(f"AUDIO: {audio_path}")
    print(f"TEMPO_BPM: {result.tempo_bpm}")
    print(f"PLUCK_CANDIDATES: {len(result.pluck_candidates)}")
    print(f"NOTES: {len(result.notes)}")
    print(f"PRINTED_EVENTS: {len(collapsed_events)}")
    print(f"GROUP_GAP_MS: {args.group_gap_ms:.1f}")
    print(f"COLLAPSE_GAP_MS: {args.collapse_gap_ms:.1f}")
    print(f"KEEP_NONE: {args.keep_none}")
    if iois.size:
        print(f"MIN_IOI_MS: {float(np.min(iois) * 1000.0):.1f}")
        print(f"MEDIAN_IOI_MS: {float(np.median(iois) * 1000.0):.1f}")
        print(f"MAX_IOI_MS: {float(np.max(iois) * 1000.0):.1f}")
        print(f"IOI_LT_120MS: {int(np.sum(iois < 0.120))}")
        print(f"IOI_LT_180MS: {int(np.sum(iois < 0.180))}")

    print()
    print("GROUP VIEW")

    printed = 0
    for group_idx, group in enumerate(groups, start=1):
        if printed >= args.limit:
            break

        tokens = [event["note_name"] for event in group]
        print(f"G{group_idx:03d} TOKENS: {' -> '.join(tokens)}")
        print("IDX  GAPms  TIME(s)  NOTE  MIDI  FREQ(Hz)  CONF  STRGTH  MERGED")

        previous_time_s = None
        for event in group:
            if printed >= args.limit:
                break

            if previous_time_s is None:
                gap_ms = "START"
            else:
                gap_ms = f"{(event['time_s'] - previous_time_s) * 1000.0:5.0f}"

            merged_suffix = (
                f"x{event['merged_count']}" if event["merged_count"] > 1 else ""
            )
            print(
                f"{event['idx']:03d}  "
                f"{gap_ms:>5s}  "
                f"{event['time_s']:7.3f}  "
                f"{event['note_name']:4s}  "
                f"{event['midi_note']:>4s}  "
                f"{event['frequency_hz']:8.2f}  "
                f"{event['confidence']:5.3f}  "
                f"{event['strength']:6.3f}  "
                f"{merged_suffix}"
            )
            previous_time_s = event["time_s"]
            printed += 1

        print()

    if len(collapsed_events) > args.limit:
        print(f"... truncated {len(collapsed_events) - args.limit} additional printed events")
