# CMI-Bench: A Comprehensive Benchmark for Evaluating Music Instruction Following

**Paper:** [CMI-BENCH: A COMPREHENSIVE BENCHMARK FOR EVALUATING MUSIC INSTRUCTION FOLLOWING](https://arxiv.org/abs/2506.12285v1)

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

Our study benchmarks 11 publicly available audio-text LLMs, covering a wide spectrum of model architectures and training paradigms.

  * Qwen2-Audio
  * Qwen-Audio
  * SALMONN-Audio
  * MusiLingo
  * LTU
  * LTU-AS
  * MU-LLaMA
  * Audio-Flamingo
  * GAMA
  * GAMA-IT
  * Pengi

## 📊 Key Findings

1.  **LLMs Underperform Supervised Baselines**: Across most tasks, instruction-following LLMs fall significantly short of task-specific supervised MIR models, except in music captioning.
2.  **Generalization is Limited**: Models perform best on datasets that were likely part of their training corpus, indicating that generalization to unseen or structurally different tasks remains a key challenge.
3.  **Sequential Tasks are Challenging**: All models struggle with tasks requiring structured, time-based outputs like melody extraction and beat tracking. This is likely due to the diversity in prompt formats and limited exposure to dense temporal supervision during training.
4.  **Emotion Regression Fails**: No model provides usable predictions for arousal and valence, highlighting a fundamental gap in mapping continuous perceptual attributes from music.
5.  **Cultural and Gender Bias**: A fine-grained analysis reveals biases toward Western instruments and pop genres. We also observed performance differences in identifying male versus female voices.

## 🛠️ Getting Started with the Toolkit

The CMI-Bench evaluation toolkit is designed for easy and standardized evaluation of audio-text LLMs on MIR tasks.

*(This is a hypothetical example of how the toolkit might be used, based on the paper's description.)*

**1. Installation**

```bash
git clone https://github.com/nicolaus625/CMI-bench
cd cmi-bench
pip install -r requirements.txt
```

**2. Prepare Your Data**

Download the required datasets and place them in the `./data` directory following the prescribed structure.

**3. Configure Your Model**

Add your model's configuration to `models.yaml`, specifying the model name, path to weights, and class for inference.

```yaml
- name: "YourAwesomeAudioLLM"
  path: "/path/to/your/weights"
  class: "YourModelWrapper"
```

**4. Run Evaluation**

Use the `evaluate.py` script to run the benchmark on a specific task or all tasks.

```bash
# Evaluate a single model on the key detection task
python evaluate.py --model YourAwesomeAudioLLM --task key_detection

# Evaluate all models on all tasks
python evaluate.py --model all --task all
```

**5. View Results**

Results will be saved in the `results/` directory in a structured format, ready for analysis.

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