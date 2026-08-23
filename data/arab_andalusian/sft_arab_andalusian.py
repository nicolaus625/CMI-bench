import argparse
import json
from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = DATASET_DIR.parents[2] / "CMI-bench-experiment" / "datasets" / "arab_andalusian"
AUDIO_DIR = Path("data") / "arab_andalusian" / "documents"

TASKS = {
    "ArabAndalusian_nawba": {
        "field": "nawba",
        "minor": "arab_andalusian_nawba",
        "instruction": (
            "Identify the nawba of this Arab-Andalusian music excerpt."
        ),
    },
    "ArabAndalusian_tab": {
        "field": "tab",
        "minor": "arab_andalusian_tab",
        "instruction": (
            "Identify the tab or modal classification of this Arab-Andalusian music excerpt."
        ),
    },
    "ArabAndalusian_form": {
        "field": "form",
        "minor": "arab_andalusian_form",
        "instruction": (
            "Identify the musical form of this Arab-Andalusian music excerpt."
        ),
    },
    "ArabAndalusian_mizan": {
        "field": "mizan",
        "minor": "arab_andalusian_mizan",
        "instruction": (
            "Identify the mizan or rhythmic cycle of this Arab-Andalusian music excerpt."
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the Arab-Andalusian dataset to CMI-Bench JSONL."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Arab-Andalusian raw dataset root (default: sibling CMI-bench-experiment checkout).",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=-1.0,
        help="Maximum seconds to use from each annotated section. The default -1 uses full sections.",
    )
    return parser.parse_args()


def parse_time(value):
    parts = [float(part) for part in value.split(":")]
    return sum(part * (60 ** idx) for idx, part in enumerate(reversed(parts)))


def source_audio_path(source_root, mbid):
    return source_root / "documents" / mbid / f"{mbid}.mp3"


def benchmark_audio_path(mbid):
    return AUDIO_DIR / mbid / f"{mbid}.mp3"


def label_value(section, field):
    value = section.get(field) or {}
    return value.get("transliterated_name") or value.get("name") or ""


def collect_labels(descriptions, field):
    return sorted(
        {
            label_value(section, field)
            for entry in descriptions
            for section in entry.get("sections", [])
            if label_value(section, field)
        }
    )


def format_instruction(task, labels):
    return (
        f"{task['instruction']}\n"
        "You must choose exactly one transliterated name from the following list:\n"
        f"{', '.join(labels)}.\n"
        "Return only the selected name. No explanation."
    )


def make_record(task_name, entry, section, section_idx, max_duration, labels):
    task = TASKS[task_name]
    field = task["field"]
    start = parse_time(section["start_time"])
    end = parse_time(section["end_time"])
    if max_duration is not None and max_duration > 0:
        end = min(end, start + max_duration)

    mbid = entry["mbid"]
    label = section[field]

    return {
        "instruction": format_instruction(task, labels),
        "input": "<|SOA|><AUDIO><|EOA|>",
        "output": label_value(section, field),
        "uuid": f"arab_andalusian_{task['minor']}_{mbid}_{section_idx:03d}",
        "split": ["test"],
        "task_type": {
            "major": ["multi-class classification"],
            "minor": [task["minor"]],
        },
        "domain": "music",
        "audio_path": [str(benchmark_audio_path(mbid))],
        "audio_start": start,
        "audio_end": end,
        "source": "Arab-Andalusian",
        "other": {
            "tag": "null",
            "mbid": mbid,
            "section_index": section_idx,
            "section_start_time": section["start_time"],
            "section_end_time": section["end_time"],
            "title": entry.get("title", ""),
            "transliterated_title": entry.get("transliterated_title", ""),
            "archive_url": entry.get("archive_url", ""),
            "musescore_url": entry.get("musescore_url", ""),
            "label_id": label.get("id", ""),
            "label_name": label.get("name", ""),
            "label_transliterated_name": label.get("transliterated_name", ""),
            "nawba": section.get("nawba", {}),
            "tab": section.get("tab", {}),
            "form": section.get("form", {}),
            "mizan": section.get("mizan", {}),
        },
    }


def write_task(task_name, descriptions, source_root, max_duration, labels):
    output_path = DATASET_DIR / f"CMI_{task_name}.jsonl"
    count = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for entry in descriptions:
            if not source_audio_path(source_root, entry["mbid"]).exists():
                skipped += len(entry.get("sections", []))
                continue
            for idx, section in enumerate(entry.get("sections", [])):
                if not label_value(section, TASKS[task_name]["field"]):
                    skipped += 1
                    continue
                record = make_record(task_name, entry, section, idx, max_duration, labels)
                if record["audio_end"] <= record["audio_start"]:
                    skipped += 1
                    continue
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    print(f"Wrote {count} samples to {output_path} (skipped {skipped})")


def main():
    args = parse_args()
    max_duration = None if args.max_duration == -1 else args.max_duration
    description_json = args.source_root / "andalusian_description.json"
    descriptions = json.load(description_json.open("r", encoding="utf-8"))
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for task_name in TASKS:
        labels = collect_labels(descriptions, TASKS[task_name]["field"])
        write_task(
            task_name,
            descriptions,
            source_root=args.source_root,
            max_duration=max_duration,
            labels=labels,
        )


if __name__ == "__main__":
    main()
