# SongFormBench for CMI-Bench

This directory contains the lightweight CMI-Bench annotation file and its
conversion script. Raw audio, section labels, extracted features, checkpoints,
and upstream utilities are intentionally excluded from Git.

The local experiment layout used by the conversion script is:

```text
../CMI-bench-experiment/datasets/songformer/
├── data/
│   ├── SongFormBench.jsonl
│   ├── audios/
│   ├── labels/
│   └── mels/
├── upstream_repository/
└── utils/
```

Generate `CMI_SongFormBench.jsonl` with:

```bash
python data/songformer/sft_songformer.py
```

For a dataset stored elsewhere, pass its root explicitly:

```bash
python data/songformer/sft_songformer.py --source-root /path/to/songformer
```

The generated records use project-relative audio paths under
`data/songformer/data/audios/`. Copy or symlink the private audio directory at
that location before running inference.
