# Model inference guide

CMI-Bench uses a registry of model adapters so that inference, resuming, sampling, and result serialization follow the same path for every supported model.

## Common setup

Create a Python environment appropriate for the model you want to run, then install the common dependencies:

```bash
pip install -r model/requirements.txt
```

Run commands from the repository root. Python 3.10 or newer is recommended.

Model checkpoints default to `~/.cache/cmi-bench/models`. Use either of these portable alternatives:

```bash
python model/infer.py --model qwen2 --models-root /path/to/models
python model/infer.py --model qwen2 --model-path /path/to/Qwen2.5-Omni-7B
```

## Supported adapters

Canonical model keys are listed below. Several adapters also provide short aliases; use `python model/infer.py --help` for the complete current list.

| Model key | Model family | Default checkpoint or API model |
| :--- | :--- | :--- |
| `qwen2` | Qwen2.5-Omni | `Qwen2.5-Omni-7B` |
| `qwen3_omni` | Qwen3-Omni | `Qwen3-Omni-30B-A3B` |
| `audio_flamingo2` | Audio Flamingo 2 | `Audio_Flamingo_2-7B` |
| `audio_flamingo3` | Audio Flamingo 3 | `Audio_Flamingo_3-7B` |
| `audio_reasoner` | Audio Reasoner | `Audio_Reasoner-7B` |
| `audsemthinker` | AudSemThinker | local AudSemThinker checkout |
| `baichuan_omni` | Baichuan-Omni | `Baichuan-Omni-11B` |
| `mellow` | Mellow | local Mellow checkout |
| `music_flamingo` | Music Flamingo | `Music_Flamingo-7B` |
| `moss_audio_8b` | MOSS-Audio | `MOSS-Audio-8B-Instruct` |
| `openomni` | OpenOmni | `OpenOmni/qwen2` |
| `gpt_audio` | OpenAI GPT Audio | `gpt-audio-1.5` |
| `gemini_3_1_pro` | Gemini 3.1 Pro | `gemini-3.1-pro-preview` |
| `gemini_3_5_flash` | Gemini 3.5 Flash | `gemini-3.5-flash` |

The implementation is under [`adapters/`](./adapters/):

* `base.py` defines the adapter interface and registry;
* `common.py` handles dataset discovery, audio paths, temporary clips, and output names;
* `hf_audio.py` contains local-model adapters;
* `api_audio.py` contains OpenAI and Gemini adapters.

## Running inference

Run one dataset:

```bash
python model/infer.py \
  --model qwen2 \
  --file-path data/GTZAN/CMI_GTZAN.jsonl \
  --output-dir model/results
```

Run every discovered instruction file:

```bash
python model/infer.py --model qwen2
```

Existing result files are resumed by default. Each successful prediction is persisted atomically unless `--checkpoint-every` is changed. Use `--overwrite` only when you intend to replace an existing result file.

Useful options:

| Option | Purpose |
| :--- | :--- |
| `--limit N` | Process at most `N` test records per dataset. |
| `--sample-plan cmi_min_n` | Apply the predefined minimum-size sampling plan. |
| `--sample-seed N` | Make sampling deterministic. |
| `--max-audio-seconds N` | Skip audio segments longer than `N` seconds. |
| `--skip-oom` | Continue after CUDA out-of-memory failures. |
| `--checkpoint-every N` | Save after every `N` successful predictions. |
| `--device-map` | Pass the device placement strategy to the model loader. |
| `--torch-dtype` | Pass the requested dtype to the model loader. |

## API models

Set credentials in the environment rather than writing them into scripts or configuration files:

```bash
export OPENAI_API_KEY=...
python model/infer.py --model gpt_audio --file-path data/GTZAN/CMI_GTZAN.jsonl

export GEMINI_API_KEY=...
python model/infer.py --model gemini_3_1_pro --file-path data/GTZAN/CMI_GTZAN.jsonl
```

API model names and retry behavior can be overridden with environment variables documented in `adapters/api_audio.py`.

## Legacy model environments

The original release also evaluated Qwen-Audio, Qwen2-Audio, SALMONN, Audio Flamingo, MusiLingo, LTU/LTU-AS, MuLLaMA, GAMA/GAMA-IT, and Pengi. Their source trees and model-specific instructions remain under `model/` and `eval/`.

These models require mutually incompatible dependency versions in some cases—for example, older Audio Flamingo code expects an older Transformers release. Use separate environments instead of installing every legacy and current model into one environment.

Checkpoint files are intentionally excluded from Git. Follow the upstream model instructions and pass their local location through `--models-root`, `--model-path`, or the adapter-specific environment variable.

## Adding another model

Create a `ModelAdapter` subclass, register it with `@register_adapter`, and implement:

```python
def load(self):
    ...
    return self

def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
    ...
```

Keep model-specific preprocessing inside the adapter and return only the final text response. The shared runner will handle dataset iteration, resuming, checkpointing, and result serialization.
