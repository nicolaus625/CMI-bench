import json
import tqdm
import os
import pandas as pd




pd_list = pd.read_csv(f"musiccaps-public.csv")
ytid_dict = pd_list.set_index('ytid')['is_audioset_eval'].to_dict()
caption_dict = pd_list.set_index('ytid')['caption'].to_dict()
train_file_list = [i for i in os.listdir(f"{PATH}/MusicCaps") if ytid_dict[i.split(".")[0]]==False and i.endswith('.wav')]
test_file_list = [i for i in os.listdir(f"{PATH}/MusicCaps") if ytid_dict[i.split(".")[0]]==True and i.endswith('.wav')]
for split in ["train", "test"]:
    data_samples = []

    for data in tqdm.tqdm(eval(f"{split}_file_list")):
        audio_path = os.path.join(f"{PATH}", f"MusicCaps/{data}")
        if not audio_path.endswith('.wav'):
            audio_path = audio_path.split(".")[0] + ".wav"
        data_sample = {
                "instruction": "Please provide the caption of the given audio.",
                "input": f"<|SOA|>{data}<|EOA|>",
                "output": caption_dict[data.split(".")[0]], 
                "uuid": "",
                "audioid": f"{data}",
                "split": [split if split != "valid" else "dev"],
                "task_type": {"major": ["global_MIR"], "minor": ["music_captioning"]},
                "domain": "music",
                "source": "Youtubet",
                "other": {}
            }
        # change uuid
        uuid_string = f"{data_sample['instruction']}#{data_sample['input']}#{data_sample['output']}"
        unique_id = hashlib.md5(uuid_string.encode()).hexdigest()[:16] #只取前16位
        if unique_id in existed_uuid_list:
            sha1_hash = hashlib.sha1(uuid_string.encode()).hexdigest()[:16] # 为了相加的时候位数对应上 # 将 MD5 和 SHA1 结果相加,并计算新的 MD5 作为最终的 UUID
            unique_id = hashlib.md5((unique_id + sha1_hash).encode()).hexdigest()[:16]
        existed_uuid_list.add(unique_id)
        data_sample["uuid"] = f"{unique_id}"

        # try to load the audio file
        data_samples.append(data_sample)
        data_samples.append(data_sample)
        data_samples.append(data_sample)
        # print(data_samples)
        # break

    # Save to JSONL format
    output_file_path = f'{PATH}/MusicCaps_3{split}.jsonl'  # Replace with the desired output path
    with open(output_file_path, 'w') as outfile:
        # for sample in data_samples:
        json.dump(data_samples, outfile)

        # outfile.write('\n')
    outfile.close()

