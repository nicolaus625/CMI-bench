import pandas as pd
import numpy as np
import json
import tqdm


classes = "bass, brass, flute, guitar, keyboard, mallet, organ, reed, string, synth_lead, vocal"

data_samples = []

for split in ["test"]:
    audio_dir = f"data/NSynth/nsynth-{split}/audio"
    metadata = json.load(open(f"nsynth-{split}/examples.json",'r'))
    metadata = json.load(open(f"nsynth-{split}/examples.json",'r'))
    metadata = [(k, v['instrument_family_str']) for k, v in metadata.items()]
    length = len(metadata)
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        audio_path = metadata[i][0]
        data_samples.append({
            "instruction": #f"What is the instrument of given audio. Please choose from {classes}",
            """Identify the primary instrument in the given audio. You must choose exactly one instrument from the following list:
Instruments: piano, guitar, violin, cello, trumpet, saxophone, flute, clarinet, drum, bass.

Your response should contain only the selected instrument.

Example 1: violin
Example 2: trumpet
Example 3: drum""",
            "input": "<|SOA|><AUDIO><|EOA|>",
            "output": metadata[i][1],
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["multi-class"], "minor": ["instrument_classification"]},
            "domain": "music",
            "audio_path": [f"{audio_dir}/{audio_path}.wav"],
            "audio_start":0.0, 
            "audio_end":-1,
            "source": "Nsynth",
            "other": {"tag":"null"}
        })
        # REMARK: the evaluation metrics should use gmean ensamble score as defined in MIREX
    
with open(f"CMI_Nsynth_instrument.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
    f.close()
