# Lyra for CMI-Bench

This directory adapts the Lyra Greek traditional and folk music dataset into
CMI-Bench JSONL tasks.

Raw audio, download caches, diagnostics, and the upstream checkout are excluded
from Git. The default local layout is:

```bash
../CMI-bench-experiment/datasets/lyra/
├── audio/
├── diagnostics/
└── upstream_repository/
```

Generate the CMI task files with:

```bash
python data/lyra/sft_lyra.py
```

Use `--source-root /path/to/lyra` if the upstream checkout is elsewhere.

Generated tasks:

- `CMI_Lyra_instrument.jsonl`
- `CMI_Lyra_genre.jsonl`
- `CMI_Lyra_place.jsonl`
- `CMI_Lyra_dance.jsonl`

To generate files that only include clips already present under
`data/lyra/audio/`, run:

```bash
python data/lyra/sft_lyra.py --split test --available-only
```

This writes files such as:

```text
CMI_Lyra_instrument_test_available.jsonl
CMI_Lyra_genre_test_available.jsonl
CMI_Lyra_place_test_available.jsonl
CMI_Lyra_dance_test_available.jsonl
```

Run inference on only these available clips with:

```bash
sbatch job_infer_lyra_available.sh
```

Lyra provides mel-spectrogram `.npy` files and YouTube timestamp metadata, but
not local audio clips for every track. CMI-Bench model inference expects audio
files, so this adapter uses the convention:

```text
data/lyra/audio/<lyra_id>.wav
```

Each wav should contain the timestamped excerpt for that Lyra id. The original
YouTube id, start/end timestamps, and mel path are preserved in each record's
`other` field.

## Download and Crop Audio

If you have the rights and permissions to retrieve the referenced YouTube audio,
you can create the wav clips with:

```bash
python data/lyra/download_audio.py --split test
```

Both scripts accept `--source-root`. The downloader writes audio and cache data
to the sibling experiment directory by default; `--audio-dir` and `--cache-dir`
override those locations.

Useful options:

```bash
# Try only a few examples first
python data/lyra/download_audio.py --split test --limit 5

# Generate both train and test clips
python data/lyra/download_audio.py --split all

# Keep the full downloaded YouTube audio cache
python data/lyra/download_audio.py --split test --keep-cache

# Use browser/exported cookies when YouTube requires them
python data/lyra/download_audio.py --split test --cookies /path/to/cookies.txt
```

The script requires `yt-dlp` and `ffmpeg` on `PATH`.
It also works when `yt-dlp` is installed as a Python package and `ffmpeg` is
provided by `imageio-ffmpeg`.

The cluster-specific job wrappers are kept in the private experiment directory.
