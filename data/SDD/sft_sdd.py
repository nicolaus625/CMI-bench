import pandas as pd
import numpy as np
import json
import tqdm


audio_dir = ""
data_samples = []

for split in ["test"]:
    metadata = pd.read_csv('song_describer.csv', usecols=['caption_id','caption','is_valid_subset','path']) 
    # filtered_metadata = metadata[metadata["is_valid_subset"] == True]

    length = len(metadata)
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        caption_id, caption, is_valid_subset, audio_path = metadata.iloc[i]
        data_samples.append({
            "instruction": "Please provide the description of given song.",
            "input": f"<|SOA|><AUDIO><|EOA|>",
            "output": caption,
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["captioning"], "minor": ["music_captioningn"]},
            "domain": "music",
            "audio_path": [f"data/SDD/audio/audio/{audio_path[:-4]}.2min.mp3"],
            "audio_start":0.0, 
            "audio_end":30.0,
            "source": "MTG-SDD",
            "other": {"tag":"null"}
        })
    
with open(f"CMI_SDD.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
