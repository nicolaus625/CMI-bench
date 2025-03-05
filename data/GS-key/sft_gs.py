import numpy as np
import json
import tqdm



    


audio_dir = "data/GS-key/giantsteps_clips/wav"
classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor"""
data_samples = []

for split in ["train", "valid", "test"]:
    with open("giantsteps_clips/meta.json") as f:
        metadata = json.load(f)
        f.close()
    
    audio_names_without_ext = [k for k in metadata.keys() if metadata[k]['split'] == split]
    length = len(audio_names_without_ext)
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        audio_path = audio_names_without_ext[i]
        data_samples.append({
            "instruction": # f"Estimate the key of the given audio. Please choose from {classes}",
           """nstruction for Musical Key Estimation
            Estimate the musical key of the given audio. You must choose exactly one key from the following options:

            C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major,
            C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor.

            Your response should only contain ONE selected key.

            Few-Shot Examples
            Example 1:
            C major

            Example 2:
            E minor

            Example 3:
            Bb major""",
            "input": "<|SOA|><AUDIO><|EOA|>",
            "output": metadata[audio_path]['y'],
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["multi-class"], "minor": ["key_detection"]},
            "domain": "music",
            
            "audio_path": [f"{audio_dir}/{audio_path}.wav"],
            "audio_start":0.0, 
            "audio_end":30.0,
            "source": "GiantSteps",
            "other": {"tag":"null"}
        })
        # REMARK: the evaluation metrics should use gmean ensamble score as defined in MIREX
    
with open(f"CMI_GS_key.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
    f.close()
