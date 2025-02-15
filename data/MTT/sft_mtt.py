import pandas as pd
import numpy as np
import json
import tqdm


audio_dir = "mp3"
labels = np.load('binary_label.npy')
tags = np.load("tags.npy")
data_samples = []

for split in ["train", "valid", "test"]:
    metadata = pd.read_csv(filepath_or_buffer=f'{split}.tsv', 
                                        sep='\t',
                                        names = ['uuid', 'audio_path'])
    length = len(metadata)
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        uuid, audio_path = metadata.iloc[i]
        index = np.nonzero(labels[uuid])[0]
        data_samples.append({
            "instruction": "Please provide the tags of given music.",
            "input": f"<|SOA|><AUDIO><|EOA|>",
            "output": ", ".join(tags[index]),
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["multi_label"], "minor": ["music_tagging"]},
            "domain": "music",
            "audio_path": [f"data/MTT/mp3/{audio_path}"],
            "audio_start":0.0, 
            "audio_end":30.0,
            "source": "The MagnaTagATune Dataset",
            "other": {"tag":"null"}
        })
    
with open(f"CMI_MTT.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
