import numpy as np
import torch
import tqdm
import json
import glob
import os


def tensor_to_tuple_string(tensor: torch.Tensor) -> str:
    """
    Convert a torch tensor of shape (n,4) to a formatted string of (start time, end time, tech) tuples.
    
    Parameters:
        tensor (torch.Tensor): Input tensor with shape (n,4). Each column represents onset_time,offset_time,IPT,note,
    
    Returns:
        str: A string representing a list of tuples with formatted (time, MIDI).
    """
    # Extract first (time) and last (MIDI) columns
    time_values = tensor[:, 0].tolist()
    end_values = tensor[:, 1].tolist()
    # print(tensor.shape, tensor[:2])
    tech_values = [technique[tech] for tech in tensor[:, 2].tolist()]
    # print(tensor[:, 2].tolist())
    # print([technique[tech] for tech in tensor[:, 2].tolist()[:10]])
    # Format each tuple with time rounded to 2 decimal places and MIDI as an integer
    formatted_tuples = [(f"{float(start):.4f}",f"{float(end):.4f}", tech) for start, end, tech in zip(time_values, end_values, tech_values) if tech!="boxian"]
    # Convert list to string
    if len(formatted_tuples) == 0:
        formatted_tuples = [("0.0", str(clip_duration), "No Tech")]
    return str(formatted_tuples)


clip_duration = 10.0
      
data_samples = []
technique = {'chanyin': "Vibrato", 
             'dianyin': "Point Note", 
             'shanghua': "Upward Portamento", 
             'xiahua': "Downward Portamento", 
             'huazhi': "Glissando", 
             'guazou': "Glissando", 
             'lianmo': "Plucks", 
             'liantuo': "Plucks", 
             'yaozhi': "Tremolo", 
             'boxian': "0"}
classes = "Vibrato, Point Note, Upward Portamento, Downward Portamento, Plucks, Glissando, Tremolo"


for split in ["train", "validation","test"]:
    # TODO: better to merge the train/valid
    track_names = [os.path.basename(i) for i in glob.glob(f"Guzheng_Tech99/data/label/{split}/*.csv")]
    label_files = [
        f"Guzheng_Tech99/data/label/{split}/{track_name}" for track_name in track_names
    ]
    audio_files = [
        f"Guzheng_Tech99/data/audio/{split}/{track_name[:-4]}.flac"
        for track_name in track_names
    ]
    for idx, label_file in tqdm.tqdm(enumerate(label_files), total=len(label_files), desc=f"Processing {split}"):
        # times_labels = torch.Tensor(np.genfromtxt(label_file, delimiter=",")) #(time,2)
        data = np.genfromtxt(label_file, delimiter=",", dtype=str)

        def convert_to_tensor(value):
            try:
                return float(value)  # Convert to float if possible
            except ValueError:
                return value  # Keep original string

        times_labels = np.vectorize(convert_to_tensor)(data)
        times_labels = times_labels[1:] # reduce the fist row/title
        # times_labels = torch.tensor(times_labels)
        # print(times_labels.shape, times_labels[:3])
        # print(times_labels[-1, 0])
        
        # time_offsets = torch.arange(
        #     0, times_labels[-1, 0] + clip_duration, clip_duration
        # ) #times_labels[-1, 0] is time ends, time_offsets = tensor([  0.,  30.,  60., ...
        time_offsets = np.arange(0, float(times_labels[-1, 0]) + clip_duration, clip_duration)
        # intervals = torch.vstack((time_offsets[:-1], time_offsets[1:])) # 0-30, 30-60, ...
        intervals = np.vstack((time_offsets[:-1], time_offsets[1:]))  # Shape (2, #intervals)
        # label_invervals = torch.logical_and(
        #     times_labels[:, :1] > intervals[0], times_labels[:, :1] < intervals[1]
        # ).T  # shape (#intervals ?, times TF inside the intervals or not )
        # print(times_labels[:, :1].shape, intervals.shape)
        label_invervals = np.logical_and(
            torch.tensor(times_labels[:, :1].astype(np.float32)).float() > torch.tensor(intervals[0, :]),
            torch.tensor(times_labels[:, :1].astype(np.float32)).float() < torch.tensor(intervals[1, :])
        ).numpy().T  # Transpose to match expected shape (#intervals, #times)
        for offset, label_interval in zip(time_offsets, label_invervals):
            data_samples.append({
                "instruction": f"Please detect the timestep of Guzheng (Chinese Kyoto) techniques of the audio. Such techniques includes:{classes}. The output format should be a list of (start timestep, end time, technique) tuples.",
                "input": f"<|SOA|><AUDIO><|EOA|>",
                "output": tensor_to_tuple_string(times_labels[label_interval]),
                "uuid": "",
                "split": [split if split != "valid" else "dev"],
                "task_type": {"major": ["seq_multi-class"], "minor": ["technique_detection"]},
                "domain": "music",
                "audio_path": ["data/Guzheng/" + audio_files[idx]],
                "audio_start": float(offset), 
                "audio_end": float(offset) + clip_duration,
                "source": "Guzheng Tech99",
                "other": {"tag":"null"}
            })
    
with open(f"CMI_Guzheng_Tech.jsonl", "w") as f:
    for data_sample in data_samples:
        f.write(json.dumps(data_sample) + "\n")
