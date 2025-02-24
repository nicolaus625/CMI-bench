import os
import numpy as np
import json
import tqdm
import pandas as pd


def quantise(beats):
   return [int(round(b * 100)) / 100 for b in beats]


def get_sample(excerpt_path, beats, split="train", key="gtzan", type_="beat"):
    data_sample = {
        "instruction": f"Identify and list the timestamps of all {type_}s in this audio track. Use the format of `0.0s,0.54s,1.0ss, ...`",
        "input": f"<|SOA|><AUDIO><|EOA|>",
        "output": ",".join([f"{b}s" for b in beats]),
        "uuid": "",
        "split": [split if split != "valid" else "dev"],
        "task_type": {"major": ["seq_multi-class"], "minor": [f"{type_}_tracking"]},
        "audio_path": [excerpt_path],
        "domain": "music",
        "audio_start":0.0, 
        "audio_end":30.0,
        "source": key,
        "other": {"tag":"null"}
    }
    return data_sample




load_annotation = np.load(f'data/full_beat_annotation.npz', allow_pickle=True)

for key in ["ballroom", "gtzan"]:  # "hainsworth", "carnetic", "smc"
    annotation = load_annotation[key]

    with open(f'data/audio_lists/{key}.txt', 'r') as f:
        audio_root = f.readlines()

    audio_root = [item.replace('\n', '') for item in audio_root]
    if key == "ballroom":
        audio_root = [f'data/ballroom/BallroomData/{item[46:]}' for item in audio_root]
    elif key == "gtzan":
        audio_root = [f'data/GTZAN/Data/genres_original/{item[43:]}' for item in audio_root]

    metadata = pd.read_csv(f'../GTZAN/test_filtered.txt', 
                        names = ['audio_path'])
    test_lists = [os.path.basename(metadata.iloc[i].item()) for i in range(len(metadata))]
    
    beat_data_samples = []
    downbeat_data_samples = []
    for idx, ann in tqdm.tqdm(enumerate(annotation)):
        # print(ann.shape, ann)
        audio_path = audio_root[idx]
        if len(ann.shape) == 1:
            beats = quantise(ann)
            downbeats = None
        elif key != "rwc":
            beats = quantise(ann[:,0])
            downbeats = quantise(ann[ann[:, 1] == 1, 0])
        
        if key =="ballroom":
            # tempo = infer_tempo(beats, fps=100)
            sample = get_sample(audio_path, beats, split="test", key=key)
            beat_data_samples.append(sample)
            down_sample = get_sample(audio_path, downbeats, split="test", key=key, type_="downbeat")
            downbeat_data_samples.append(down_sample)          
        elif key == "gtzan":
            if "jazz.00054" in audio_path:
                continue
            if os.path.basename(audio_path) not in test_lists:
                continue
            sample = get_sample(audio_path, beats, split="test", key=key)
            beat_data_samples.append(sample)
            if downbeats:
                down_sample = get_sample(audio_path, downbeats, split="test", key=key, type_="downbeat")
                downbeat_data_samples.append(down_sample)                    
        
            
    with open(f'CMI_{key}_beat.jsonl', 'w') as f:
        # for sample in data_samples:
        for data_sample in beat_data_samples:
            f.write(json.dumps(data_sample) + "\n")
    
    f.close()
    
    with open(f'CMI_{key}_downbeat.jsonl', 'w') as f:
        # for sample in data_samples:
        for data_sample in downbeat_data_samples:
            f.write(json.dumps(data_sample) + "\n")
        
    f.close()
