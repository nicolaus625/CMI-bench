import pandas as pd
import numpy as np
import json
import tqdm



audio_dir = "data/GTZAN/Data/genres_original"
data_samples = []

for split in ["train", "valid", "test"]:
    metadata = pd.read_csv(f'{split}_filtered.txt', 
                        names = ['audio_path'])
    length = len(metadata)
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        audio_path = metadata.iloc[i].item()
        # print(audio_path)
        data_samples.append({
            "instruction": "Which of the following genres does the given music blong to? blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock.",
            "input": f"<|SOA|><AUDIO><|EOA|>",
            "output": audio_path.split("/")[0] ,
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["multi_class"], "minor": ["genre_classification"]},
            "domain": "music",
            "audio_path": [f"{audio_dir}/{audio_path}"],
            "audio_start":0.0, 
            "audio_end":30.0,
            "source": "GTZAN Dataset",
            "other": {"tag":"null"}
        })
    
with open(f"CMI_GTZAN.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
