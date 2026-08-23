import argparse
import csv
import sys
import subprocess
from pathlib import Path
from shutil import which


DATASET_DIR = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT_ROOT = DATASET_DIR.parents[2] / "CMI-bench-experiment" / "datasets" / "lyra"
DEFAULT_SOURCE_ROOT = DEFAULT_EXPERIMENT_ROOT / "upstream_repository"
DEFAULT_AUDIO_DIR = DEFAULT_EXPERIMENT_ROOT / "audio"
DEFAULT_CACHE_DIR = DEFAULT_EXPERIMENT_ROOT / ".cache" / "youtube"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Lyra YouTube audio and crop timestamped clips."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Lyra upstream checkout (default: sibling CMI-bench-experiment checkout).",
    )
    parser.add_argument(
        "--split",
        choices=("training", "test", "all"),
        default="test",
        help="Which Lyra split to process.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help="Directory where cropped wav clips are written.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory where full downloaded YouTube audio files are cached.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clips.")
    parser.add_argument("--keep-cache", action="store_true", help="Keep full downloaded audio files.")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Output wav sample rate.")
    parser.add_argument(
        "--failed-log",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT / "diagnostics" / "download_failed.tsv",
        help="Path where failed rows are logged.",
    )
    parser.add_argument(
        "--format",
        default="bestaudio/best",
        help="yt-dlp format selector.",
    )
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Optional cookies.txt file for yt-dlp.",
    )
    return parser.parse_args()


def require_tool(name):
    path = which(name)
    if path is None and name == "ffmpeg":
        try:
            import imageio_ffmpeg

            path = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            path = None
    if path is None:
        if name == "yt-dlp":
            try:
                import yt_dlp  # noqa: F401

                return [sys.executable, "-m", "yt_dlp"]
            except Exception:
                pass
        raise RuntimeError(
            f"Required tool '{name}' was not found. Install '{name}' or activate an "
            f"environment that provides it, then rerun this script."
        )
    return path


def read_rows(source_root, split):
    splits = ("training", "test") if split == "all" else (split,)
    for split_name in splits:
        split_path = source_root / "data" / "split" / f"{split_name}.tsv"
        with split_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                row["_split"] = split_name
                yield row


def run_command(command):
    subprocess.run(command, check=True)


def download_youtube_audio(row, cache_dir, yt_dlp, audio_format, cookies=None):
    youtube_id = row["youtube-id"]
    output_template = cache_dir / f"{youtube_id}.%(ext)s"
    existing = sorted(
        path for path in cache_dir.glob(f"{youtube_id}.*") if not path.name.endswith(".part")
    )
    if existing:
        return existing[0]

    command = list(yt_dlp) if isinstance(yt_dlp, list) else [yt_dlp]
    command.extend([
        "--no-playlist",
        "--format",
        audio_format,
        "--output",
        str(output_template),
    ])
    if cookies is not None:
        command.extend(["--cookies", str(cookies)])
    command.append(f"https://www.youtube.com/watch?v={youtube_id}")
    run_command(command)

    downloaded = sorted(
        path for path in cache_dir.glob(f"{youtube_id}.*") if not path.name.endswith(".part")
    )
    if not downloaded:
        raise FileNotFoundError(f"yt-dlp finished but no cached audio found for {youtube_id}")
    return downloaded[0]


def crop_clip(row, source_audio, output_path, ffmpeg, sample_rate):
    start = float(row["start-ts"])
    end = float(row["end-ts"])
    duration = max(0.0, end - start)
    if duration <= 0:
        raise ValueError(f"Invalid timestamps for {row['id']}: start={start}, end={end}")

    temp_path = output_path.with_suffix(".tmp.wav")
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(temp_path),
    ]
    run_command(command)
    temp_path.replace(output_path)


def main():
    args = parse_args()
    yt_dlp = require_tool("yt-dlp")
    ffmpeg = require_tool("ffmpeg")

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    skipped = 0
    failed = 0
    failures = []
    for row in read_rows(args.source_root, args.split):
        if args.limit is not None and total >= args.limit:
            break
        total += 1

        output_path = args.audio_dir / f"{row['id']}.wav"
        if output_path.exists() and not args.overwrite:
            skipped += 1
            print(f"[skip] {output_path}")
            continue

        try:
            print(f"[download] {row['id']} ({row['youtube-id']})")
            source_audio = download_youtube_audio(
                row,
                args.cache_dir,
                yt_dlp,
                args.format,
                cookies=args.cookies,
            )
            print(f"[crop] {row['id']} -> {output_path}")
            crop_clip(row, source_audio, output_path, ffmpeg, args.sample_rate)
            if not args.keep_cache:
                source_audio.unlink(missing_ok=True)
        except Exception as exc:
            failed += 1
            failures.append((row, str(exc)))
            print(f"[fail] {row['id']}: {exc}")

    print(f"Done. processed={total} skipped={skipped} failed={failed}")
    if failed:
        args.failed_log.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_log.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["id", "youtube-id", "split", "start-ts", "end-ts", "error"])
            for row, error in failures:
                writer.writerow([
                    row["id"],
                    row["youtube-id"],
                    row["_split"],
                    row["start-ts"],
                    row["end-ts"],
                    error,
                ])
        print(f"Failed rows written to {args.failed_log}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
