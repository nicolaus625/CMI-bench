import json
import tqdm
import os
import pandas as pd




pd_list = pd.read_csv(f"musiccaps-public.csv")
ytid_dict = pd_list.set_index('ytid')['is_audioset_eval'].to_dict()
caption_dict = pd_list.set_index('ytid')['caption'].to_dict()
train_file_list = [i for i in os.listdir(f"audio") if ytid_dict[i.split(".")[0]]==False and i.endswith('.wav')]
test_file_list = [i for i in os.listdir(f"audio") if ytid_dict[i.split(".")[0]]==True and i.endswith('.wav')]
data_samples = []
for split in ["train", "test"]:
    for data in tqdm.tqdm(eval(f"{split}_file_list")):
        audio_path = os.path.join(f"audio/{data}")
        if not audio_path.endswith('.wav'):
            audio_path = audio_path.split(".")[0] + ".wav"
        data_samples.append({
                "instruction": "Please provide the caption of the given audio.",
                "input": "<|SOA|><AUDIO><|EOA|>",
                "output": caption_dict[data.split(".")[0]], 
                "uuid": "",
                "split": [split if split != "valid" else "dev"],
                "task_type": {"major": ["captioning"], "minor": ["music_captioning"]},
                "domain": "music",
                "audio_path": ["data/MusicCaps/audio/" + data],
                "audio_start":0.0, 
                "audio_end":30.0,
                "source": "AudioSet",
                "other": {"tag":"null"}
            })

with open(f"CMI_VocalSet_tech.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
    f.close()
