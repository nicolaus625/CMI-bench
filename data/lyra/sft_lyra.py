import csv
import json
import argparse
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = DATASET_DIR.parents[2] / "CMI-bench-experiment" / "datasets" / "lyra" / "upstream_repository"
AUDIO_DIR = Path("data") / "lyra" / "audio"
MEL_DIR = Path("data") / "lyra" / "mel-spectrograms"

TASKS = {
    "Lyra_instrument": {
        "field": "instruments",
        "major": "multi-label classification",
        "minor": "lyra_instrument",
        "instruction": (
            "Identify the musical instruments present in the given Greek traditional "
            "or folk music excerpt."
        ),
    },
    "Lyra_genre": {
        "field": "genres",
        "major": "multi-label classification",
        "minor": "lyra_genre",
        "instruction": (
            "Identify the genre tags of the given Greek traditional or folk music excerpt."
        ),
    },
    "Lyra_place": {
        "field": "place",
        "major": "multi-label classification",
        "minor": "lyra_place",
        "instruction": (
            "Identify the geographic place or region associated with the given Greek "
            "traditional or folk music excerpt."
        ),
    },
    "Lyra_dance": {
        "field": "is-danced",
        "major": "binary classification",
        "minor": "lyra_dance",
        "instruction": (
            "Determine whether the given Greek traditional or folk music excerpt is "
            "intended for dancing. Return only yes or no."
        ),
    },
}


def split_values(value):
    if value in ("", "None", None):
        return []
    return [item.strip() for item in value.split("|") if item.strip()]


def format_output(task_name, row):
    if task_name == "Lyra_dance":
        return "yes" if row["is-danced"] == "1" else "no"
    return ", ".join(split_values(row[TASKS[task_name]["field"]]))


def collect_labels(task_name, rows):
    if task_name == "Lyra_dance":
        return ["yes", "no"]
    field = TASKS[task_name]["field"]
    return sorted({label for row in rows for label in split_values(row[field])})


def format_instruction(task_name, labels):
    task = TASKS[task_name]
    if task_name == "Lyra_dance":
        return task["instruction"]
    return (
        f"{task['instruction']}\n"
        "Select all applicable names from the following list:\n"
        f"{', '.join(labels)}.\n"
        "Return only the selected names separated by commas. No explanation."
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert Lyra TSV splits to CMI-Bench JSONL.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Lyra upstream checkout (default: sibling CMI-bench-experiment checkout).",
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Only include rows whose wav file exists under data/lyra/audio/.",
    )
    parser.add_argument(
        "--split",
        choices=("training", "test", "all"),
        default="all",
        help="Which split to write.",
    )
    return parser.parse_args()


def read_split(source_root, split_name):
    path = source_root / "data" / "split" / f"{split_name}.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def make_record(task_name, row, split, labels, source_root):
    task = TASKS[task_name]
    start_ts = float(row["start-ts"])
    end_ts = float(row["end-ts"])
    track_id = row["id"]

    return {
        "instruction": format_instruction(task_name, labels),
        "input": "<|SOA|><AUDIO><|EOA|>",
        "output": format_output(task_name, row),
        "uuid": f"lyra_{task['minor']}_{track_id}",
        "split": [split],
        "task_type": {
            "major": [task["major"]],
            "minor": [task["minor"]],
        },
        "domain": "music",
        "audio_path": [str(AUDIO_DIR / f"{track_id}.wav")],
        "audio_start": 0.0,
        "audio_end": max(0.0, end_ts - start_ts),
        "source": "Lyra",
        "other": {
            "tag": "null",
            "lyra_id": track_id,
            "youtube_id": row["youtube-id"],
            "youtube_start_ts": row["start-ts"],
            "youtube_end_ts": row["end-ts"],
            "mel_path": str(MEL_DIR / f"{track_id}.npy"),
            "instruments": split_values(row["instruments"]),
            "genres": split_values(row["genres"]),
            "place": split_values(row["place"]),
            "coordinates": split_values(row["coordinates"]),
            "is_danced": row["is-danced"],
        },
    }


def audio_exists(row):
    return (DATASET_DIR / "audio" / f"{row['id']}.wav").exists()


def iter_selected_rows(source_root, split, available_only):
    split_pairs = (("training", "train"), ("test", "test")) if split == "all" else (
        (split, "train" if split == "training" else split),
    )
    for split_file, split_name in split_pairs:
        for row in read_split(source_root, split_file):
            if available_only and not audio_exists(row):
                continue
            yield row, split_name


def write_task(task_name, labels, source_root, split="all", available_only=False):
    suffix = "_available" if available_only else ""
    split_suffix = "" if split == "all" else f"_{split}"
    output_path = DATASET_DIR / f"CMI_{task_name}{split_suffix}{suffix}.jsonl"
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row, split_name in iter_selected_rows(source_root, split, available_only):
            handle.write(json.dumps(make_record(task_name, row, split_name, labels, source_root), ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} samples to {output_path}")


def main():
    args = parse_args()
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = list(read_split(args.source_root, "training")) + list(read_split(args.source_root, "test"))
    for task_name in TASKS:
        labels = collect_labels(task_name, all_rows)
        write_task(
            task_name,
            labels,
            source_root=args.source_root,
            split=args.split,
            available_only=args.available_only,
        )


if __name__ == "__main__":
    main()
