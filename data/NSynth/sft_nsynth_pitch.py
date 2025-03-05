import pandas as pd
import numpy as np
import json
import tqdm



data_samples = []

for split in ["test"]:
    audio_dir = f"data/NSynth/nsynth-{split}/audio"
    metadata = json.load(open(f"nsynth-{split}/examples.json",'r'))
    metadata = [(k, v['pitch']) for k, v in metadata.items()]
    length = len(metadata)
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        audio_path = metadata[i][0]
        data_samples.append({
            "instruction": #f"What is the pitch of given audio. Note pitches are represented as in the MIDI specification, using integers from 0 (lowest pitch) to 127 (highest pitch) for your answer. While MIDI has a total of 128 pitches (i.e., 10 octaves), the lowest pitch in this dataset is 9 (A_1), and the highest pitch is 119 (C flat 9).",
            """Identify the pitch of the given audio. You must choose exactly one pitch from the given range.

Pitches are represented using the MIDI standard, where each pitch is assigned an integer value between 0 (lowest) and 127 (highest). In this dataset, the lowest pitch is 9 (1 octave below A_0, 13.75Hz), and the highest pitch is 119 (B 8, 7902.13Hz). Be sure to convert a musical note to its MIDI number:
Middle C (C4) = 60
A440 (A4) = 69
One octave higher = +12 MIDI numbers
One octave lower = -12 MIDI numbers

Your response should contain only one MIDI pitch number.

Example 1, when the note is A1: 33
Example 2, when the note is F5: 77
Example 3, when the note is A4: 69""",
            "input": "<|SOA|><AUDIO><|EOA|>",
            "output": metadata[i][1],
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["multi-class"], "minor": ["pitch_classification"]},
            "domain": "music",
            "audio_path": [f"{audio_dir}/{audio_path}.wav"],
            "audio_start":0.0, 
            "audio_end":-1,
            "source": "Nsynth",
            "other": {"tag":"null"}
        })
        # REMARK: the evaluation metrics should use gmean ensamble score as defined in MIREX
    
with open(f"CMI_Nsynth_pitch.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
    f.close()
