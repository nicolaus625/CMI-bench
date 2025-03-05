import pandas as pd
import numpy as np
import json
import tqdm



classes = 'belt, breathy, inhaled, lip_trill, spoken, straight, trill, trillo, vibrato, vocal_fry'

audio_dir = "data/VocalSet/audio"

data_samples = []

for split in ["train", "valid", "test"]:
    metadata = pd.read_csv(filepath_or_buffer=f'{split}_t.txt', names = ['audio_path'])
    
    length = len(metadata)
    
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        audio_path = metadata.iloc[i][0]
        data_samples.append({
            "instruction": #f"What is the singing technique of the given audio. Please choose from {classes}",
            """
Identify the singing technique used in the given audio. You must choose exactly one from the following options:
Singing Techniques: belt, breathy, inhaled, lip_trill, spoken, straight, trill, trillo, vibrato, vocal_fry.

Your response should contain only ONE selected technique.

Example 1: belt
Example 2: breathy
Example 3: vibrato""", 
            "input": "<|SOA|><AUDIO><|EOA|>",
            "output": audio_path.split('/')[0],
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["multi-class"], "minor": ["singing_technique_classification"]},
            "domain": "music",
            "audio_path": [f"{audio_dir}/{audio_path}"],
            "audio_start":0.0, 
            "audio_end":-1,
            "source": "VocalSet",
            "other": {"tag":"null"}
        })
        # REMARK: the evaluation metrics should use gmean ensamble score as defined in MIREX
    
with open(f"CMI_VocalSet_tech.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
    f.close()
