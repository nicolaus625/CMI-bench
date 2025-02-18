import json
import tqdm


PATH = "mtg-jamendo-dataset/data"
# Load your dataset

def get_top50tags(_="test"):
    data_samples = []
    for split in [_]:
        input_file_path = f'{PATH}/splits/split-0/autotagging_top50tags-{split}.tsv'  # Replace with the actual path to your dataset

        with open(input_file_path, "r") as f:
            for idx, line in tqdm.tqdm(enumerate(f), desc=f"Processing {split}"):
                if idx > 0:
                    tmp = line.strip().split("\t")
                    top50tagss = tmp[5:]
                    audio_path = tmp[3]
                    audio_path = audio_path[:-4] + ".low.mp3"
                                    
                    data_sample = {
                        "instruction": "Please provide the music tags of given audio? This may include but not limited to instruments, emotions and genres",
                        "input": f"<|SOA|><AUDIO><|EOA|>",
                        "output": ", ".join(sorted([top50tags.split("---")[-1] for top50tags in top50tagss])),
                        "uuid": "",
                        "audio_path": [f"data/MTG/audio-low/{audio_path}"],
                        "audio_start":0.0, 
                        "audio_end":30.0,
                        "split": [split if split != "validation" else "dev"],
                        "task_type": {"major": ["multi_label"], "minor": ["top50tags_classification"]},
                        "domain": "music",
                        "source": "MTG-Jamendo",
                        "other": {"tag":"null"}
                    }            

                    data_samples.append(data_sample)
                # if idx > 2:
                #     break
        f.close()
            
    return data_samples

if __name__ == "__main__":
    print("start")
    
    output_file_path = "CMI_MTG_top50tags.jsonl"
    with open(output_file_path, 'w') as outfile:
        for split in ["test", "train", "validation"]:
            data_samples = get_top50tags(split)
            # Save to JSONL format
            # output_file_path = f'top50tags_{split}.jsonl'
            for sample in data_samples:
                outfile.write(json.dumps(sample)  + "\n")
    outfile.close()

