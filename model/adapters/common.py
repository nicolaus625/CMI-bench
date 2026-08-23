from __future__ import annotations

import glob
import json
import os
import random
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = PROJECT_ROOT / "model"
DATA_ROOT = PROJECT_ROOT / "data"
EXTERNAL_DATA_ROOT = os.environ.get("CMI_BENCH_DATA_ROOT")
HF_PATH = Path(
    os.environ.get("CMI_BENCH_MODEL_DIR", Path.home() / ".cache" / "cmi-bench" / "models")
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_audio_path(audio_relative_path: str) -> str:
    candidates = [PROJECT_ROOT / audio_relative_path]
    if EXTERNAL_DATA_ROOT:
        external_root = Path(EXTERNAL_DATA_ROOT)
        candidates.append(external_root / audio_relative_path)
    if EXTERNAL_DATA_ROOT and audio_relative_path.startswith("data/"):
        candidates.append(
            Path(EXTERNAL_DATA_ROOT) / audio_relative_path.replace("data/", "testdata/", 1)
        )

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    tried = [str(candidate) for candidate in candidates]
    raise FileNotFoundError(f"Audio file not found for '{audio_relative_path}'. Tried: {tried}")


def list_dataset_files(file_paths: Iterable[str] | None = None) -> List[str]:
    if file_paths:
        resolved_paths = []
        for path in file_paths:
            candidate = Path(path)
            if candidate.exists():
                resolved_paths.append(str(candidate))
                continue

            project_candidate = PROJECT_ROOT / path
            if project_candidate.exists():
                resolved_paths.append(str(project_candidate))
                continue

            raise FileNotFoundError(
                f"Dataset file not found for '{path}'. Tried: {candidate}, {project_candidate}"
            )
        return sorted(resolved_paths)
    return sorted(glob.glob(str(DATA_ROOT / "*" / "CMI*.jsonl")))


def load_test_records(
    file_path: str,
    limit: int | None = None,
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> List[dict]:
    records = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"][0] != "test":
                continue
            records.append(record)
    if sample_size is not None and sample_size > 0 and len(records) > sample_size:
        rng = random.Random(sample_seed)
        records = rng.sample(records, sample_size)
    if limit is not None:
        records = records[:limit]
    return records


def load_audio(*args, **kwargs):
    from data_loader import load_audio as _load_audio

    return _load_audio(*args, **kwargs)


@contextmanager
def clipped_audio_file(
    source_audio_path: str,
    target_sr: int,
    start: float,
    end: float,
    prefix: str,
) -> Iterator[str]:
    import torchaudio
    from data_loader import load_audio

    waveform = load_audio(source_audio_path, target_sr=target_sr, start=start, end=end)
    with tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False) as handle:
        temp_path = handle.name
    try:
        torchaudio.save(temp_path, waveform, target_sr)
        yield temp_path
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def normalize_generation_text(text: str) -> str:
    text = text.strip()
    if "<RESPONSE>" in text:
        match = re.search(r"<RESPONSE>\s*(.*?)\s*(?:</RESPONSE>|$)", text, flags=re.DOTALL)
        if match:
            text = match.group(1).strip()
    text = re.sub(r"<THINK>.*?</THINK>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    return text.strip()


def build_output_path(output_root: str, model_name: str, dataset_file: str) -> str:
    output_dir = Path(output_root) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name = os.path.basename(dataset_file)
    if dataset_name.startswith("CMI_"):
        dataset_name = dataset_name[4:]
    return str(output_dir / f"{model_name}_{dataset_name}")
