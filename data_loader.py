import os
import json
import random
import glob
import tqdm

import torch
from torch.utils.data import Dataset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio


RESAMPLE_RATE = 24000
DATASET_ROOT = "/import/c4dm-datasets-ext/yinghao_tmp/CMI-bench/"


def load_audio(
    file_path, 
    target_sr, 
    is_mono=True, 
    is_normalize=False,
    # crop_to_length_in_sec=None, 
    # crop_to_length_in_sample_points=None, 
    # crop_randomly=False, 
    pad=False,
    return_start=False,
    device=torch.device('cpu'),
    start=0.0,
    end=30.0,
):
    """Load audio file and convert to target sample rate.
    Supports cropping and padding.

    Args:
        file_path (str): path to audio file
        target_sr (int): target sample rate, if not equal to sample rate of audio file, resample to target_sr
        is_mono (bool, optional): convert to mono. Defaults to True.
        is_normalize (bool, optional): normalize to [-1, 1]. Defaults to False.
        crop_to_length_in_sec (float, optional): crop to specified length in seconds. Defaults to None.
        crop_to_length_in_sample_points (int, optional): crop to specified length in sample points. Defaults to None. Note that the crop length in sample points is calculated before resampling.
        crop_randomly (bool, optional): crop randomly. Defaults to False.
        pad (bool, optional): pad to specified length if waveform is shorter than specified length. Defaults to False.
        device (torch.device, optional): device to use for resampling. Defaults to torch.device('cpu').
        start (float, optional): start time in seconds. Defaults to 0.0.
        end (float, optional): end time in seconds. Defaults to 30.0.
        
    
    Returns:
        torch.Tensor: waveform of shape (1, n_sample)
    """
    # TODO: deal with target_depth
    # print(file_path)
    try:
        waveform, sample_rate = torchaudio.load(file_path)
    except Exception as e:
        waveform, sample_rate = torchaudio.backend.soundfile_backend.load(file_path)
    if waveform.shape[0] > 1:
        if is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if is_normalize:
        waveform = waveform / waveform.abs().max()
    
    # waveform, start = crop_audio(
    #     waveform, 
    #     sample_rate, 
    #     crop_to_length_in_sec=crop_to_length_in_sec, 
    #     crop_to_length_in_sample_points=crop_to_length_in_sample_points, 
    #     crop_randomly=crop_randomly, 
    #     pad=pad,
    # )
    if end == -1:
        crop_duration_in_sample = waveform.shape[-1] - start * sample_rate
    else:
        crop_duration_in_sample = int((end - start) * sample_rate)
    start = int(start * sample_rate)
    if waveform.shape[-1] > start + crop_duration_in_sample:
        waveform = waveform[..., start:start + crop_duration_in_sample]
    elif waveform.shape[-1] < start + crop_duration_in_sample:
        waveform = waveform[..., start:]
        if pad:
            waveform = torch.nn.functional.pad(waveform, (0, crop_duration_in_sample - waveform.shape[-1]))
    
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
        waveform = waveform.to(device)
        resampler = resampler.to(device)
        waveform = resampler(waveform)
    
    if return_start:
        return waveform, start
    return waveform


def crop_audio(
    waveform, 
    sample_rate, 
    crop_to_length_in_sec=None, 
    crop_to_length_in_sample_points=None, 
    crop_randomly=False, 
    pad=False,
):
    """Crop waveform to specified length in seconds or sample points.
    Supports random cropping and padding.

    Args:
        waveform (torch.Tensor): waveform of shape (1, n_sample)
        sample_rate (int): sample rate of waveform
        crop_to_length_in_sec (float, optional): crop to specified length in seconds. Defaults to None.
        crop_to_length_in_sample_points (int, optional): crop to specified length in sample points. Defaults to None.
        crop_randomly (bool, optional): crop randomly. Defaults to False.
        pad (bool, optional): pad to specified length if waveform is shorter than specified length. Defaults to False.

    Returns:
        torch.Tensor: cropped waveform
        int: start index of cropped waveform in original waveform
    """
    assert crop_to_length_in_sec is None or crop_to_length_in_sample_points is None, \
    "Only one of crop_to_length_in_sec and crop_to_length_in_sample_points can be specified"

    # convert crop length to sample points
    crop_duration_in_sample = None
    if crop_to_length_in_sec:
        crop_duration_in_sample = int(sample_rate * crop_to_length_in_sec)
    elif crop_to_length_in_sample_points:
        crop_duration_in_sample = crop_to_length_in_sample_points

    # crop
    start = 0
    if crop_duration_in_sample:
        if waveform.shape[-1] > crop_duration_in_sample:
            if crop_randomly:
                start = random.randint(0, waveform.shape[-1] - crop_duration_in_sample)
            waveform = waveform[..., start:start + crop_duration_in_sample]

        elif waveform.shape[-1] < crop_duration_in_sample:
            if pad:
                waveform = torch.nn.functional.pad(waveform, (0, crop_duration_in_sample - waveform.shape[-1]))
    
    return waveform, start


class CMIDataset(Dataset):
    def __init__(self, ann_path, audio_root_path=DATASET_ROOT, split="train", sr=24000):
        super().__init__()
        self.path = audio_root_path
        try:
            self.annotation = json.load(open(ann_path, "r"))
        except:
            with open(ann_path, "r") as file:
                self.annotation = file.read()
                self.annotation = [ eval(tmp) for tmp in self.annotation.split("\n")[:-1]]
        if split == "valid":
            split = "dev"
        self.annotation = [ann for ann in self.annotation if ann["split"][0] == split]
        self.split = split
        self.sr = sr

    def __len__(self):
        return len(self.annotation)

    def collater(self, samples):
        # TODO: need adapting to multiple audio files
        raw_wav = [s["raw_wav"][0] for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"][0]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)

        text = [s["text"] for s in samples]
        task = [s["task"] for s in samples]
        Q = [s["Q"] for s in samples]
        id = [s["id"] for s in samples]

        return {
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": text,  # output, answer
            "task": task,
            "Q": Q,  # instruction, question
            "id": id,
        }

    def __getitem__(self, index):
        ann = self.annotation[index]
        audio_list = []
        for audio_relative_path in ann["audio_path"]:
            audio_path = os.path.join(self.path, audio_relative_path)
            # print(audio_path, ann["audio_start"], ann["audio_end"])
            audio = load_audio(
                    audio_path,
                    target_sr=self.sr,
                    is_mono=True,
                    is_normalize=True,
                    pad=False,
                    start=ann["audio_start"],
                    end=ann["audio_end"],
                )
            audio = audio.squeeze(0)
            audio_list.append(audio)
        
        return {
            "raw_wav": audio_list,
            "text": ann["output"],
            "task": f'{ann["task_type"]["minor"][0]}',
            "Q": ann["instruction"],
            "id": ann["uuid"],
        }
        
    def sr_update(self, sr):
        self.sr = sr
        

class MultipleDataset(Dataset):
    def __init__(self, ann_paths, split="train", sr=24000):
        super().__init__()
        self.jsonl_list = glob.glob(f"{ann_paths}/*/CMI_*.jsonl")
        dataset_list = []
        for ann_path in self.jsonl_list:
            dataset_list.append(CMIDataset(ann_path, split=split)) 
        self.dataset = ConcatDataset(dataset_list)
        self.sr = sr
        self.split = split
    
    def __len__(self):
        return len(self.dataset)
    
    def collater(self, samples):
        raw_wav = [s["raw_wav"][0] for s in samples]
        raw_wav_length = torch.tensor([len(s["raw_wav"][0]) for s in samples])
        raw_wav = pad_sequence(raw_wav, batch_first=True, padding_value=0)
        paddding_mask = torch.arange(raw_wav.size(1)).unsqueeze(0) >= raw_wav_length.unsqueeze(1)
        
        return {
            "raw_wav": raw_wav,
            "padding_mask": paddding_mask,
            "text": [s["text"] for s in samples],
            "task": [s["task"] for s in samples],
            "Q": [s["Q"] for s in samples],
            "id": [s["id"] for s in samples],
        }
    
    def __getitem__(self, index):
        return self.dataset[index] 
    
    def sr_update(self, sr):
        dataset_list = [ CMIDataset(ann_path, split=self.split, sr=sr) for ann_path in self.jsonl_list ]
        self.dataset = ConcatDataset(dataset_list)
   
 
if __name__ == "__main__":
    dataset = CMIDataset("data/DSing/CMI_DSing.jsonl", split="test")
    print(len(dataset))
    dataset.sr_update(44100)
    for idx, data in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if idx < 6:
            print(dataset[idx])
            # break
            
    # dataset = MultipleDataset("data")
    # dataset.sr_update(44100)
    # print(len(dataset))
    # print(dataset[0])
