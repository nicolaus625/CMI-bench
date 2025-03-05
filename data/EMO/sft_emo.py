import numpy as np
import json
import tqdm


audio_dir = "data/EMO/emomusic/wav"
data_samples_arousal = []
data_samples_valence = []

for split in ["train", "valid", "test"]:
    with open("emomusic/meta.json") as f:
        metadata = json.load(f)
        f.close()

    audio_names_without_ext = [k for k in metadata.keys() if metadata[k]['split'] == split]
    length = len(audio_names_without_ext)
    for i in tqdm.tqdm(range(length), desc=f"Processing {split}"):
        audio_path = audio_names_without_ext[i]
        data_samples_arousal.append({
            "instruction": #"Estimate the arousal score of the given music on a scale from 1 to 9, where 1 represents the lowest arousal level and 9 represents the highest.",
            """Estimate the arousal score of the given music on a scale from 1 to 9, where 1 represents the lowest arousal level (calm, relaxing) and 9 represents the highest arousal level (energetic, intense). Provide a numerical estimate based on your perception of the music.

            Example 1, when the music is slow and gentle, suggesting a low arousal level:
            Estimated score: 2.

            Example 2, when the music is energetic and driving, indicating a high arousal level:
            Estimated score: 8.

            Example 3, when the music has a balanced tempo and moderate intensity:
            Estimated score: 5.""",
            
            "input": "<|SOA|><AUDIO><|EOA|>",
            "output": str(metadata[audio_path]["y"][0]),
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["regression"], "minor": ["emotion_regression"]},
            "domain": "music",
            "audio_path": [f"{audio_dir}/{audio_path}.wav"],
            "audio_start":0.0, 
            "audio_end":45.0,
            "source": "EMO",
            "other": {"tag":"null"}
        })
        data_samples_valence.append({
            "instruction": #"Estimate the valence score of the given music on a scale from 1 to 9, where 1 represents the lowest arousal level and 9 represents the highest.",
            """Estimate the valence score of the given music on a scale from 1 to 9, where 1 represents the lowest valence level (sad, melancholic) and 9 represents the highest valence level (happy, cheerful). Provide a numerical estimate based on your perception of the music’s emotional positivity.

            Few-Shot Examples:
            Example 1, when the music is slow and somber, conveying a sense of sadness:
            Estimated score: 2.

            Example 2, when the music is bright and uplifting, evoking happiness and positivity:
            Estimated score: 8.

            Example 3, when the music has a neutral or mixed emotional quality, neither too sad nor too happy:
            Estimated score: 5.""",
            
            "input": "<|SOA|><AUDIO><|EOA|>",
            "output": str(metadata[audio_path]["y"][1]),
            "uuid": "",
            "split": [split if split != "valid" else "dev"],
            "task_type": {"major": ["regression"], "minor": ["emotion_regression"]},
            "domain": "music",
            "audio_path": [f"{audio_dir}/{audio_path}.wav"],
            "audio_start":0.0, 
            "audio_end":45.0,
            "source": "EMO",
            "other": {"tag":"null"}
        })
        # REMARK: the evaluation metrics should nomalise the output of each model performance and calculate the R2 
    
with open(f"CMI_EMO_arousal.jsonl", "w") as f:
    for data_sample in data_samples_arousal:
        f.write(json.dumps(data_sample) + "\n")
    f.close()
    
with open(f"CMI_EMO_valence.jsonl", "w") as f:
    for data_sample in data_samples_arousal:
        f.write(json.dumps(data_sample) + "\n")
    f.close()
