import pandas as pd
import numpy as np
import json
import tqdm


audio_dir = ""
data_samples = []


metadata = pd.read_csv('Kaldi-Dsing-task/DSing Preconstructed/test.csv', usecols=['recording','start','end','text']) 
    # filtered_metadata = metadata[metadata["is_valid_subset"] == True]

length = len(metadata)
for i in tqdm.tqdm(range(length), desc=f"Processing test"):
    audio_path, start, end, text = metadata.iloc[i]
    data_samples.append({
        "instruction": "Please transcribe the lyrics of the given song.",
        "input": f"<|SOA|><AUDIO><|EOA|>",
        "output": text,
        "uuid": "",
        "split": ["test"],
        "task_type": {"major": ["transcription"], "minor": ["lyrics_transcription"]},
        "domain": "music",
        "audio_path": [f"data/DSing/sing_300x30x2/{audio_path}"],
        "audio_start":start, 
        "audio_end":end,
        "source": "DSing",
        "other": {"tag":"null"}
    })

with open(f"CMI_DSing.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
