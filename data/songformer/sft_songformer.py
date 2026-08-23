import argparse
import json
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = DATASET_DIR.parents[2] / "CMI-bench-experiment" / "datasets" / "songformer"
OUTPUT_JSONL = DATASET_DIR / "CMI_SongFormBench.jsonl"

INSTRUCTION = """Analyze the song structure of the given audio track.
Identify each structural section and return a Python string representation of a list of tuples in the format (start time second, end time second, section label).
Use section labels such as intro, verse, pre-chorus, chorus, bridge, inst, outro, and silence.
Return only the list, with no extra explanation."""


def project_relative_path(songformer_path):
    """Convert SongFormBench-local paths to CMI-bench project-relative paths."""
    if songformer_path is None:
        return ""
    return str(Path("data") / "songformer" / songformer_path)


def metadata_value(value):
    return "" if value is None else value


def labels_to_segments(labels, duration):
    segments = []
    sorted_labels = sorted(labels, key=lambda item: float(item["start"]))

    for idx, segment in enumerate(sorted_labels):
        label = segment["label"]
        if label == "end":
            continue

        start = float(segment["start"])
        if idx + 1 < len(sorted_labels):
            end = float(sorted_labels[idx + 1]["start"])
        else:
            end = float(duration)

        if end <= start:
            continue

        segments.append((start, end, label))

    return segments


def format_segments(segments):
    return str(
        [
            (f"{start:.4f}", f"{end:.4f}", label)
            for start, end, label in segments
        ]
    )


def convert_entry(entry):
    segments = labels_to_segments(entry["labels"], entry["duration"])

    return {
        "instruction": INSTRUCTION,
        "input": "<|SOA|><AUDIO><|EOA|>",
        "output": format_segments(segments),
        "uuid": entry["id"],
        "split": ["test"],
        "task_type": {
            "major": ["seq_multi-class"],
            "minor": ["song_structure_analysis"],
        },
        "domain": "music",
        "audio_path": [project_relative_path(entry["audio_path"])],
        "audio_start": 0.0,
        "audio_end": float(entry["duration"]),
        "source": "SongFormBench",
        "other": {
            "tag": "null",
            "subset": metadata_value(entry.get("subset", "")),
            "language": metadata_value(entry.get("language", "")),
            "youtube_url": metadata_value(entry.get("youtube_url", "")),
            "sample_rate": metadata_value(entry.get("sample_rate", "")),
            "label_path": project_relative_path(entry.get("label_path", "")),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Convert SongFormBench metadata to CMI-Bench JSONL.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="SongFormBench raw dataset root (default: sibling CMI-bench-experiment checkout).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_jsonl = args.source_root / "data" / "SongFormBench.jsonl"
    data_samples = []
    with raw_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data_samples.append(convert_entry(json.loads(line)))

    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for sample in data_samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Wrote {len(data_samples)} samples to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
