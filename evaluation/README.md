# TaBee Evaluation Workflow

This folder contains a small, repeatable workflow for filling the missing
algorithmic result tables in `Report.tex`.

## 1. Select samples

Use 3 small groups so the report tables have real values:

- `Isolated notes`: controlled single-note recordings.
- `Short bass phrases`: short local recordings such as `cli/test.wav` and `cli/test2.wav`.
- `Bass-only song excerpts`: longer bass-only audio, ideally trimmed to a manageable excerpt.

For a final report, 3-5 clips per group is enough if time is limited. More is better, but a small
annotated set is much stronger than TODO tables.

## 2. Create ground truth

Copy `ground_truth_template.csv` to a new file, for example:

```powershell
Copy-Item evaluation\ground_truth_template.csv evaluation\ground_truth.csv
```

Fill one row per real note event. Use seconds for `onset_s`, MIDI note numbers for
`midi_number`, and allowed tab positions as `string:fret` pairs separated by `|`.

Example:

```csv
sample_group,sample_name,prediction_json,onset_s,midi_number,allowed_positions
Short bass phrases,test.wav,logs/test-smoke.tab.json,0.125,40,4:12|3:7|2:2
```

If exact fingering is known, write one position. If several fingerings are musically acceptable,
list them all.

## 3. Generate prediction JSON files

Run the existing processor for each sample:

```powershell
python cli\audio_to_tab.py cli\test.wav --json-out logs\test-smoke.tab.json --ascii-out logs\test-smoke.tab.txt
```

If Python reports that `librosa` is missing, install the project requirements in the environment
you use for audio processing:

```powershell
python -m pip install -r requirements.txt
```

## 4. Compute metrics

Run:

```powershell
python evaluation\evaluate_tab_json.py evaluation\ground_truth.csv --onset-tolerance 0.08
```

The script prints Markdown tables for:

- onset precision, recall, and F1
- pitch accuracy and octave errors
- exact/equivalent string-fret accuracy
- unplayable note rate

Paste those values into the TODO tables in `Report.tex`.

