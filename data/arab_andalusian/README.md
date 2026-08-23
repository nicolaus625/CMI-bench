# Arab-Andalusian for CMI-Bench

This directory contains lightweight CMI-Bench annotations and a converter for
the Arab-Andalusian dataset. The 6.4 GB source dataset is excluded from Git and
stored locally as:

```text
../CMI-bench-experiment/datasets/arab_andalusian/
```

into CMI-Bench JSONL files. The source dataset already contains local mp3
recordings under `documents/<mbid>/<mbid>.mp3`, so no YouTube download step is
needed.

Generate the CMI files with:

```bash
python data/arab_andalusian/sft_arab_andalusian.py
```

For a source dataset stored elsewhere:

```bash
python data/arab_andalusian/sft_arab_andalusian.py --source-root /path/to/ArabAndalusianDataset
```

Generated tasks:

- `CMI_ArabAndalusian_nawba.jsonl`
- `CMI_ArabAndalusian_tab.jsonl`
- `CMI_ArabAndalusian_form.jsonl`
- `CMI_ArabAndalusian_mizan.jsonl`

Each example is an annotated section from `andalusian_description.json`. By
default the converter uses the full annotated section. To cap each section at
30 seconds:

```bash
python data/arab_andalusian/sft_arab_andalusian.py --max-duration 30
```

Generated records use project-relative paths under
`data/arab_andalusian/documents/`. Copy or symlink the private `documents`
directory there before inference. For example, with the sibling experiment
layout:

```bash
ln -s ../../../CMI-bench-experiment/datasets/arab_andalusian/documents \
  data/arab_andalusian/documents
```
