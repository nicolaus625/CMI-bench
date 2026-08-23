import argparse
import json
import os
from pathlib import Path

from tqdm import tqdm

from adapters import MODEL_ADAPTERS, get_adapter_class
from adapters.common import HF_PATH, build_output_path, list_dataset_files, load_test_records, resolve_audio_path


CMI_MIN_N_SAMPLE_PLAN = {
    "GS_key": 500,
    "MTT": 300,
    "MTG_instrument": 750,
    "MTG_genre": 750,
    "MTG_emotion": 2500,
    "MTG_top50tags": 500,
    "Nsynth_instrument": 50,
    "Nsynth_pitch": 2000,
    "VocalSet_tech": 200,
    "SDD": 30,
}

CMI_CLOSED_ALL_SAMPLE_PLAN = {
    # Original sampled-min-N tasks.
    "GS_key": 500,
    "MTT": 300,
    "MTG_instrument": 750,
    "MTG_genre": 750,
    "MTG_emotion": 2500,
    "MTG_top50tags": 500,
    "Nsynth_instrument": 100,
    "Nsynth_pitch": 2000,
    "VocalSet_tech": 200,
    "SDD": 100,
    # Additional closed-model tasks.
    "EMO_valence": 125,
    "EMO_arousal": 100,
    "GTZAN": 100,
    "Guzheng_Tech": 94,
    "MedleyDB": 500,
    "MusicCaps": 100,
    "DSing": 200,
    "ballroom_beat": 100,
    "ballroom_downbeat": 100,
    "gtzan_beat": 100,
    "gtzan_downbeat": 100,
    "SongFormBench": 300,
    "ArabAndalusian_form": 300,
    "ArabAndalusian_mizan": 200,
    "ArabAndalusian_nawba": 200,
    "ArabAndalusian_tab": 200,
    "Lyra_instrument": 200,
    "Lyra_genre": 100,
    "Lyra_place": 270,
    "Lyra_dance": 200,
}

CMI_CLOSED_REMAINING_SAMPLE_PLAN = {
    "EMO_valence": 125,
    "EMO_arousal": 100,
    "GTZAN": 100,
    "Guzheng_Tech": 94,
    "MedleyDB": 500,
    "MusicCaps": 100,
    "DSing": 200,
    "ballroom_beat": 100,
    "ballroom_downbeat": 100,
    "gtzan_beat": 100,
    "gtzan_downbeat": 100,
    "SongFormBench": 300,
    "ArabAndalusian_form": 300,
    "ArabAndalusian_mizan": 200,
    "ArabAndalusian_nawba": 200,
    "ArabAndalusian_tab": 200,
    "Lyra_instrument": 200,
    "Lyra_genre": 100,
    "Lyra_place": 270,
    "Lyra_dance": 200,
}

SAMPLE_PLANS = {
    "cmi_min_n": CMI_MIN_N_SAMPLE_PLAN,
    "cmi_closed_all": CMI_CLOSED_ALL_SAMPLE_PLAN,
    "cmi_closed_remaining": CMI_CLOSED_REMAINING_SAMPLE_PLAN,
}


def infer_task_from_dataset_file(dataset_file: str, sample_plan: dict[str, int] | None = None) -> str | None:
    plan = sample_plan or CMI_MIN_N_SAMPLE_PLAN
    stem = Path(dataset_file).stem
    if stem.startswith("CMI_"):
        stem = stem[4:]
    for task in sorted(plan, key=len, reverse=True):
        if stem == task or stem.startswith(f"{task}_"):
            return task
    return None


def should_skip_dataset_for_sample_plan(dataset_file: str, task: str, sample_plan_name: str) -> bool:
    if sample_plan_name in {"cmi_closed_all", "cmi_closed_remaining"} and task.startswith("Lyra_"):
        return not Path(dataset_file).name.endswith("_test_available.jsonl")
    return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=sorted(MODEL_ADAPTERS))
    parser.add_argument("--model-path", type=str, default=None, help="Local path to the model checkpoint")
    parser.add_argument("--models-root", type=str, default=str(HF_PATH))
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory where per-model prediction files will be written",
    )
    parser.add_argument(
        "--file-path",
        nargs="*",
        default=None,
        help="Optional one or more dataset jsonl files. If omitted, run all CMI*.jsonl files under data/",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of test examples per dataset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--torch-dtype", type=str, default="auto")
    parser.add_argument(
        "--skip-oom",
        action="store_true",
        help="Skip samples that raise CUDA OOM instead of aborting the whole run",
    )
    parser.add_argument(
        "--max-audio-seconds",
        type=float,
        default=float(os.environ.get("MAX_AUDIO_SECONDS", "300")),
        help="Skip audio segments longer than this many seconds. Set <=0 to disable.",
    )
    parser.add_argument(
        "--sample-plan",
        choices=("none", *sorted(SAMPLE_PLANS)),
        default="none",
        help="Apply a named per-task sampling plan before inference.",
    )
    parser.add_argument(
        "--sample-min-floor",
        type=int,
        default=100,
        help="When using --sample-plan cmi_min_n, raise plan values below this floor.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="Seed for deterministic random sampling.",
    )
    parser.add_argument(
        "--include-unplanned",
        action="store_true",
        help="With a named --sample-plan, also run datasets not listed in the plan.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Write partial results after this many successful predictions. Use 1 to persist every sample.",
    )
    return parser.parse_args()


def _is_too_long_audio(start: float, end: float, max_audio_seconds: float | None) -> bool:
    if max_audio_seconds is None or max_audio_seconds <= 0:
        return False
    return end - start > max_audio_seconds


def _result_key_from_values(audioid, audio_start, audio_end, uuid="") -> tuple[str, str, str, str]:
    return (str(uuid or ""), str(audioid), str(audio_start), str(audio_end))


def _record_key(record: dict, source_audio: str) -> tuple[str, str, str, str]:
    return _result_key_from_values(
        source_audio,
        record["audio_start"],
        record["audio_end"],
        record.get("uuid", ""),
    )


def _result_key(result: dict) -> tuple[str, str, str, str]:
    return _result_key_from_values(
        result.get("audioid", ""),
        result.get("audio_start", ""),
        result.get("audio_end", ""),
        result.get("uuid", ""),
    )


def _load_existing_results(output_path: str, overwrite: bool = False) -> list[dict]:
    if overwrite or not Path(output_path).exists():
        return []
    try:
        with open(output_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError:
        backup_path = f"{output_path}.broken"
        os.replace(output_path, backup_path)
        print(f"Existing result JSON is broken; moved it to {backup_path}")
        return []
    if not isinstance(data, list):
        backup_path = f"{output_path}.invalid"
        os.replace(output_path, backup_path)
        print(f"Existing result JSON is not a list; moved it to {backup_path}")
        return []
    return data


def _write_results_atomic(output_path: str, results: list[dict]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


def infer_dataset(
    adapter,
    dataset_file: str,
    output_path: str,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
    max_audio_seconds: float | None = None,
    skip_oom: bool = False,
    overwrite: bool = False,
    checkpoint_every: int = 1,
):
    records = load_test_records(
        dataset_file,
        limit=limit,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )
    results = _load_existing_results(output_path, overwrite=overwrite)
    completed = {_result_key(result) for result in results}
    skipped_multi_audio = 0
    skipped_failed_audio = 0
    skipped_long_audio = 0
    skipped_existing = 0
    since_checkpoint = 0

    for record in tqdm(records, desc=Path(dataset_file).stem):
        if len(record["audio_path"]) != 1:
            skipped_multi_audio += 1
            continue

        source_audio = record["audio_path"][0]
        audio_start = record["audio_start"]
        audio_end = record["audio_end"]
        try:
            source_audio = resolve_audio_path(source_audio)
            record_key = _record_key(record, source_audio)
            if record_key in completed:
                skipped_existing += 1
                continue
            if _is_too_long_audio(audio_start, audio_end, max_audio_seconds):
                skipped_long_audio += 1
                print(
                    "Skipping long audio sample: "
                    f"{source_audio} [{audio_start}, {audio_end}] "
                    f"duration={audio_end - audio_start:g}s > {max_audio_seconds:g}s"
                )
                continue
            response = adapter.predict(
                prompt=record["instruction"],
                audio_path=source_audio,
                start=audio_start,
                end=audio_end,
            )
        except Exception as exc:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                if not skip_oom:
                    raise
                torch.cuda.empty_cache()
            skipped_failed_audio += 1
            print(
                "Skipping failed sample: "
                f"{source_audio} [{audio_start}, {audio_end}]: "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        results.append(
            {
                "question": record["instruction"],
                "response": response,
                "correct_answer": record["output"],
                "uuid": record.get("uuid", ""),
                "audioid": source_audio,
                "audio_start": audio_start,
                "audio_end": audio_end,
                "other": "",
            }
        )
        completed.add(record_key)
        since_checkpoint += 1
        if checkpoint_every > 0 and since_checkpoint >= checkpoint_every:
            _write_results_atomic(output_path, results)
            since_checkpoint = 0

    _write_results_atomic(output_path, results)

    return len(results), skipped_multi_audio, skipped_failed_audio, skipped_long_audio, skipped_existing


def main():
    args = parse_args()
    adapter_cls = get_adapter_class(args.model)
    if args.model_path:
        model_path = args.model_path
    elif getattr(adapter_cls, "is_api_model", False):
        model_path = adapter_cls.default_model_subdir
    else:
        model_path = str(Path(args.models_root) / adapter_cls.default_model_subdir)
    adapter = adapter_cls(model_path=model_path, device_map=args.device_map, torch_dtype=args.torch_dtype).load()

    dataset_files = list_dataset_files(args.file_path)
    sample_sizes = {}
    if args.sample_plan != "none":
        plan = SAMPLE_PLANS[args.sample_plan]
        planned_files = []
        for dataset_file in dataset_files:
            task = infer_task_from_dataset_file(dataset_file, plan)
            if task is None:
                if args.include_unplanned:
                    planned_files.append(dataset_file)
                continue
            if should_skip_dataset_for_sample_plan(dataset_file, task, args.sample_plan):
                print(f"Skipping {dataset_file} for sample plan {args.sample_plan}")
                continue
            planned_files.append(dataset_file)
            sample_size = plan[task]
            if args.sample_plan == "cmi_min_n":
                sample_size = max(sample_size, args.sample_min_floor)
            sample_sizes[dataset_file] = sample_size
        dataset_files = planned_files
    print(f"Found {len(dataset_files)} dataset files")
    if sample_sizes:
        print("Sampling plan:")
        for dataset_file in dataset_files:
            if dataset_file in sample_sizes:
                print(f"  {Path(dataset_file).name}: {sample_sizes[dataset_file]}")

    for dataset_file in dataset_files:
        output_path = build_output_path(args.output_dir, args.model, dataset_file)

        print(f"Processing {dataset_file}")
        done, skipped, failed, skipped_long, skipped_existing = infer_dataset(
            adapter,
            dataset_file,
            output_path,
            limit=args.limit,
            sample_size=sample_sizes.get(dataset_file),
            sample_seed=args.sample_seed,
            max_audio_seconds=args.max_audio_seconds,
            skip_oom=args.skip_oom,
            overwrite=args.overwrite,
            checkpoint_every=args.checkpoint_every,
        )
        print(f"Saved {done} predictions to {output_path}")
        if skipped_existing:
            print(f"Skipped {skipped_existing} already-completed samples in {dataset_file}")
        if skipped_long:
            print(f"Skipped {skipped_long} audio samples longer than {args.max_audio_seconds:g}s in {dataset_file}")
        if skipped:
            print(f"Skipped {skipped} multi-audio samples in {dataset_file}")
        if failed:
            print(f"Skipped {failed} failed audio samples in {dataset_file}")


if __name__ == "__main__":
    main()
