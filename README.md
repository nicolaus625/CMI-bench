# CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following


[![PWC](https://img.shields.io/badge/%F0%9F%93%8E%20arXiv-Paper-red)](https://arxiv.org/abs/2506.12285v1)
[![PWC](https://img.shields.io/badge/HuggingFace-Demo-Green)](https://huggingface.co/datasets/nicolaus625/CMI-bench/)

**Authors:** Yinghao Ma, Siyou Li, Juntao Yu, Emmanouil Benetos, Akira Maezawa

-----

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

## 🤖 Models Evaluated

Here is a revised version of the README section that improves clarity, structure, and consistency with the accompanying table:

---

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

To install model-specific environments (e.g., Qwen-audio, Qwen2-audio, Audio-Flamingo, Mu-LLaMA, MusiLingo, LTU, LTU-AS), please refer to:

📄 [`CMI-bench/model/README.md`](./model/README.md)

Each model has its own setup instructions and pre-trained checkpoints.

### 🛠️ **1. Prepare the Dataset**

#### 🛠️ **1.1 Download Test Audio**

Download test-set audio from Hugging Face:

```bash
wget https://huggingface.co/datasets/nicolaus625/CMI-bench/resolve/main/test_Data.zip
unzip test_Data.zip -d CMI-bench/data
```

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
  --output-file results
```

This command will:

* Load the specified model
* Process each input audio and instruction under `~/CMI-bench/data/*/CMI*.jsonl`
* Save predictions to `model/results/{model}/{model}_{task}.jsonl`

Available models:
`qwen`, `qwen2`, `salmonn`, `musilingo`, `ltu`, `ltu_as`, `mullama`, `flamingo`, etc.

### 🛠️ **3. Configure Your Own Model**

To add your own model:

1. Extend `infer.py` with a new `--model` option.
2. Implement a `get_{model_name}_pred()` function that takes:

   * `text` (instruction)
   * `audio_path` (test audio path)
   * any required processors or tokenizers
3. Place output JSONL results in `model/results/{model}/`.

### 🛠️ **4. Run Evaluation**

To evaluate model outputs using task-specific metrics:

```bash
python evaluate.py \
  --model qwen2 \
  --task ballroom_beat
```

You can replace `--task` with:

* A specific dataset (e.g., `GTZAN`, `MusicCaps`, `MTG_emotion`)
* Or `--task all` to run evaluation for all available tasks

Results include metrics like:

* ROC-AUC / PR-AUC (for multi-label tasks)
* WER / CER (for lyrics transcription)
* Accuracy (for multi-class classification )
* R² (for emotion regression)
* F1 (for structured outputs like beat tracking or techinique detection)
* BLEU / BERTScore (for music captioning)

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