# CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following


[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/abs/2506.12285v1)
[![PWC](https://img.shields.io/badge/HuggingFace-Demo-Green)](https://huggingface.co/datasets/nicolaus625/CMI-bench/)

**Authors:** Yinghao Ma, Siyou Li, Juntao Yu, Emmanouil Benetos, Akira Maezawa

🎉🎉🎉 Paper accepted by the 26th conference of the International Society for Music Information Retrieval (ISMIR). See you in Daejeon, Korea from September 21-25, 2025.

-----

## Development Branch

This branch extends the original ISMIR release with:

* three additional benchmarks: SongFormBench, Lyra, and Arab-Andalusian;
* nine new evaluation targets covering song structure and culturally diverse music attributes;
* a modular model-adapter interface for recent open and API-based audio-language models;
* resumable inference, atomic checkpoints, deterministic sampling plans, and long-audio filtering;
* expanded task-specific evaluation for the new datasets.

Large audio files, acoustic features, checkpoints, inference outputs, scheduler logs, and manuscript artifacts are intentionally excluded from Git. Download or generate them locally as described in the dataset-specific READMEs.

## Abstract

Recent advances in audio-text large language models (LLMs) have opened new possibilities for music understanding and generation. However, existing benchmarks are limited in scope, often relying on simplified tasks or multi-choice evaluations that fail to reflect the complexity of real-world music analysis. We introduce **CMI-Bench**, a comprehensive music instruction-following benchmark designed to evaluate audio-text LLMs on a diverse set of music information retrieval (MIR) tasks. CMI-Bench reinterprets a broad range of traditional MIR annotations into an instruction-following format and uses standardized evaluation metrics consistent with state-of-the-art MIR models. Our experiments reveal significant performance gaps between current LLMs and specialized supervised models, as well as cultural, chronological, and gender biases. CMI-Bench establishes a unified foundation for evaluating and advancing music-aware LLMs.

## 🚀 Key Contributions

  * **Comprehensive Task Coverage**: CMI-Bench includes 14 diverse MIR tasks, moving beyond simple classification to include regression, captioning, and complex sequential tasks.
  * **Standardized Evaluation**: Unlike previous benchmarks that rely on multiple-choice questions, CMI-Bench employs open-ended, task-specific metrics aligned with the MIR literature (e.g., using `mir_eval`), allowing for direct comparison with traditional supervised models.
  * **Evaluation Toolkit**: We provide a full evaluation toolkit that supports all major open-source audio-textual LLMs, enabling standardized and reproducible benchmarking.
  * **In-depth Analysis**: The benchmark facilitates a deeper analysis of model capabilities, including generalization, prompt sensitivity, and biases related to culture and gender.

## 🎵 Tasks and Datasets

CMI-Bench encompasses 14 tasks evaluated across 20 different datasets, covering a wide range of challenges in music information retrieval.

| Task | Dataset(s) | Metric(s) |
| :--- | :--- | :--- |
| **Genre Classification** | MTG-Genre, GTZAN | ROC-AUC, PR-AUC, Accuracy |
| **Emotion Tagging** | MTG-Emotion | ROC-AUC, PR-AUC |
| **Emotion Regression** | EMO | $R^2$ |
| **Instrument Classification**| MTG-Instrument, Nsynth-Instrument | ROC-AUC, PR-AUC, Accuracy |
| **Music Tagging** | MagnaTagATune, MTG-Top50 | ROC-AUC, PR-AUC |
| **Pitch Estimation** | Nsynth-Pitch | Accuracy |
| **Key Detection** | GiantSteps | Gmean Score |
| **Lyrics Transcription** | DSing | WER, CER |
| **Music Captioning** | SDD, MusicCaps | BLEU, METEOR, ROUGE, Bert-Score |
| **Melody Extraction** | MedleyDB v2 | Melody Accuracy |
| **(Down)Beat Tracking** | GTZAN-Rhythm, Ballroom | F-measure |
| **Vocal Technique** | VocalSet | Accuracy |
| **Performance Technique** | GuZheng 99 | Frame-level micro/macro-F1 |

*This is a summary of the tasks listed in Table 1 of the paper.*

### Extended benchmarks in this branch

| Benchmark | Evaluation target(s) | Metric(s) |
| :--- | :--- | :--- |
| **SongFormBench** | Song structure analysis | Boundary F1, pairwise F1, Rand index, adjusted Rand index, boundary deviation |
| **Lyra** | Instrument, genre, place, dance | Micro/macro-F1, exact match, accuracy |
| **Arab-Andalusian** | Nawba, tab, form, mizan | Accuracy |

The corresponding instruction files and preparation notes are under [`data/songformer`](./data/songformer/), [`data/lyra`](./data/lyra/), and [`data/arab_andalusian`](./data/arab_andalusian/).

## 🤖 Models Evaluated

### Evaluated Models

We benchmark **11 publicly available audio-text large language models (LLMs)**, representing a diverse range of architectures and training paradigms. These models vary in scale, input modality coverage (sound, speech, music), and design choices across encoders and decoders.

A summary of each evaluated model’s capabilities is shown below:

| Model                      | #Params | Sound | Music | Speech | 
| -------------------------- | ------- | ----- | ----- | ------ | 
| Pengi                      | 323M    | ✓     | ✓     | ✗      |  
| Audio-Flamingo             | 2.2B    | ✓     | ✓     | ✗      |  
| LTU                        | 7B      | ✓     | ✓     | ✗      |  
| LTU-AS                     | 7B      | ✓     | ✓     | ✓      |  
| MusiLingo-long             | 7B      | ✗     | ✓     | ✗      |  
| MuLLaMA                    | 7B      | ✗     | ✓     | ✗      |  
| GAMA                       | 7B      | ✓     | ✓     | ✗      |  
| GAMA-IT                    | 7B      | ✓     | ✓     | ✗      |  
| Qwen-Audio-Chat            | 8.4B    | ✓     | ✗     | ✗      |  
| Qwen2-Audio-Instruct       | 8.4B    | ✓     | ✓     | ✓      |  
| SALMONN-Audio              | 13B     | ✓     | ✓     | ✓      |  

> Note: "Sound" refers to general non-speech audio; "Music" and "Speech" indicate support for those modalities in both input understanding and reasoning tasks.

### Inference adapters in this branch

The refactored inference runner currently includes adapters for:

* Qwen2.5-Omni and Qwen3-Omni;
* Audio Flamingo 2 and 3, Music Flamingo, Audio Reasoner, and AudSemThinker;
* Baichuan-Omni, MOSS-Audio-8B, Mellow, and OpenOmni;
* OpenAI GPT Audio, Gemini 3.1 Pro, and Gemini 3.5 Flash APIs.

Run `python model/infer.py --help` in the configured model environment to see the exact model keys and aliases.

## 📊 Key Findings

1.  **LLMs Underperform Supervised Baselines**: Across most tasks, instruction-following LLMs fall significantly short of task-specific supervised MIR models, except in music captioning.
2.  **Generalization is Limited**: Models perform best on datasets that were likely part of their training corpus, indicating that generalization to unseen or structurally different tasks remains a key challenge.
3.  **Sequential Tasks are Challenging**: All models struggle with tasks requiring structured, time-based outputs like melody extraction and beat tracking. This is likely due to the diversity in prompt formats and limited exposure to dense temporal supervision during training.
4.  **Emotion Regression Fails**: No model provides usable predictions for arousal and valence, highlighting a fundamental gap in mapping continuous perceptual attributes from music.
5.  **Cultural and Gender Bias**: A fine-grained analysis reveals biases toward Western instruments and pop genres. We also observed performance differences in identifying male versus female voices.

## 🛠️ Getting Started with the Toolkit

The CMI-Bench evaluation toolkit is designed for easy and standardized evaluation of audio-text LLMs on MIR tasks.
This section guides you through preparing datasets, running inference with audio-text LLMs, and evaluating results using the **CMI-Bench** toolkit.

### 🛠️ **0. Installation**

Install the common adapter dependencies with:

```bash
pip install -r model/requirements.txt
```

Some local models require their own environment or repository-specific dependencies. See the [model inference guide](./model/README.md) and the model directories under [`model/`](./model/).

### 🛠️ **1. Prepare the Dataset**

#### 🛠️ **1.1 Download Test Audio**

Download test-set audio from Hugging Face:

```bash
wget https://huggingface.co/datasets/nicolaus625/CMI-bench/resolve/main/test_Data.zip
unzip test_Data.zip -d CMI-bench/data
```

The extended datasets have separate acquisition instructions because their audio and derived features are not stored in this Git repository:

* [Lyra preparation](./data/lyra/README.md)
* [SongFormBench preparation](./data/songformer/README.md)
* [Arab-Andalusian preparation](./data/arab_andalusian/README.md)

The raw assets for these three additions are intentionally kept outside Git in
a sibling experiment workspace:

```text
../CMI-bench-experiment/datasets/
├── songformer/       # audio, labels, features, checkpoints, upstream utilities
├── lyra/             # cropped audio, cache, diagnostics, upstream checkout
└── arab_andalusian/  # source metadata and recordings
```

The checked-in CMI JSONL files use project-relative paths. Copy or symlink the
corresponding private asset directories into `data/` before inference.

#### 🛠️ **1.2 Generate JSONL Annotation Files**

To create instruction-following data pairs in `.jsonl` format:

```bash
# Example: Generate beat tracking data
python CMI-bench/data/Beat-Transformer/sft_beat.py
```

This creates files like:

```
CMI-bench/data/Beat-Transformer/CMI_ballroom_beat.jsonl
```

Repeat similarly for other tasks by running `sft_*.py` scripts in `CMI-bench/data/*/`.

### 🛠️ **2. Inference the Model**

Run inference using:

```bash
python model/infer.py \
  --model qwen2 \
  --file-path data/GTZAN/CMI_GTZAN.jsonl \
  --output-dir model/results
```

This command will:

* load the selected adapter and checkpoint;
* process the test records in the selected instruction file;
* save predictions to `model/results/{model}/{model}_{task}.jsonl`;
* resume from existing predictions and write atomic checkpoints.

Omit `--file-path` to discover all `data/*/CMI*.jsonl` files. Useful controls include:

```bash
# Quick smoke test
python model/infer.py --model qwen2 --limit 10

# Deterministic task-specific sampling
python model/infer.py --model qwen2 --sample-plan cmi_min_n --sample-seed 0

# Replace existing result files instead of resuming
python model/infer.py --model qwen2 --overwrite
```

Local checkpoint lookup defaults to `~/.cache/cmi-bench/models`; override it with `--models-root`, set `CMI_BENCH_MODEL_DIR`, or pass a checkpoint directly with `--model-path`. API adapters read credentials from their standard environment variables, such as `OPENAI_API_KEY` and `GEMINI_API_KEY`.

### 🛠️ **3. Configure Your Own Model**

To add a model:

1. Subclass `ModelAdapter` from [`model/adapters/base.py`](./model/adapters/base.py).
2. Register the class with `@register_adapter` and give it a unique `model_key`.
3. Implement `load()` and `predict(prompt, audio_path, start, end)`.
4. Add the adapter module to [`model/adapters/__init__.py`](./model/adapters/__init__.py) if it is defined in a new file.

The CLI model choices are populated automatically from the adapter registry.

### 🛠️ **4. Run Evaluation**

To evaluate model outputs using task-specific metrics:

```bash
python evaluate.py \
  --model qwen2 \
  --task ballroom_beat
```

You can replace `--task` with:

* a specific dataset or target, such as `GTZAN`, `MusicCaps`, `MTG_emotion`, `SongFormBench`, `Lyra_genre`, or `ArabAndalusian_nawba`;
* Or `--task all` to run evaluation for all available tasks

Results include metrics like:

* ROC-AUC / PR-AUC (for multi-label tasks)
* WER / CER (for lyrics transcription)
* Accuracy (for multi-class classification )
* R² (for emotion regression)
* F1 (for structured outputs like beat tracking or technique detection)
* BLEU / BERTScore (for music captioning)
* Boundary, pairwise, and clustering measures (for song structure analysis)

## 📜 Citation

If you use CMI-Bench in your research, please cite our paper:

```bibtex
@misc{ma2025cmibenchcomprehensivebenchmarkevaluating,
      title={CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following}, 
      author={Yinghao Ma and Siyou Li and Juntao Yu and Emmanouil Benetos and Akira Maezawa},
      year={2025},
      eprint={2506.12285},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2506.12285}, 
}
```

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
