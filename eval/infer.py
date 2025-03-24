import glob
from tqdm import tqdm
import json
import os
import argparse

proj_path = "/import/c4dm-04/siyoul/CMI-bench"
cache_path = "/import/c4dm-04/siyoul/CMI-bench/cache"
jsonl_list = glob.glob(os.path.join(proj_path, "data/*/CMI*.jsonl"))

parser = argparse.ArgumentParser()
    
parser.add_argument('--output-file', default="results", type=str, help='the path to save the output json file')

parser.add_argument('--model', default="ltu", type=str, 
                    choices=["ltu", "gama", "gama_it", "ltu_as", "pengi"], 
                    help='the model to use for inference')
parser.add_argument('--device', default="0", type=str, help='the device to use for inference')
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = "cuda"



if args.model == "ltu":
    from eval.ltu import LTUModel
    eval_mdl_path = os.path.join(proj_path ,"pretrained_models/LTU/ltu_ori_paper.bin")
    base_model = os.path.join(proj_path ,"pretrained_models/vicuna-7b-v1.1")
    log_save_path = "./inference_log/"
    model = LTUModel(base_model, eval_mdl_path, log_save_path, device)
elif args.model == "gama":
    from eval.gama import GAMAModel
    eval_mdl_path = os.path.join(proj_path ,"pretrained_models/GAMA/stage4_ckpt/pytorch_model.bin")
    base_model = os.path.join(proj_path ,"pretrained_models/GAMA/Llama-2-7b-chat-hf-qformer/")
    log_save_path = "./inference_log/"
    model = GAMAModel(base_model, eval_mdl_path, log_save_path, device)
elif args.model == "gama_it":
    from eval.gama_it import GAMA_ITModel
    eval_mdl_path = os.path.join(proj_path ,"pretrained_models/GAMA-IT/stage5_ckpt/pytorch_model.bin")
    base_model = os.path.join(proj_path ,"pretrained_models/GAMA-IT/Llama-2-7b-chat-hf-qformer/")
    log_save_path = "./inference_log/"
    model = GAMA_ITModel(base_model, eval_mdl_path, log_save_path, device)
elif args.model == "ltu_as":
    from eval.ltu_as import LTU_ASModel
    eval_mdl_path = os.path.join(proj_path ,"pretrained_models/LTU-AS/ltuas_long_noqa_a6.bin")
    base_model = os.path.join(proj_path ,"pretrained_models/vicuna-7b-v1.1")
    whisper_checkpoint_path = os.path.join(proj_path ,"pretrained_models/LTU-AS/large-v1.pt")
    log_save_path = "./inference_log/"
    model = LTU_ASModel(base_model, eval_mdl_path, whisper_checkpoint_path, log_save_path, device)

elif args.model == "pengi":
    from eval.pengi import PengiModel
    model = PengiModel()

import torch
import torchaudio

def split_and_save_audio(
    file_path, 
    is_mono=True, 
    is_normalize=False,
    pad=False,
    return_start=False,
    device=torch.device(device),
    start=0.0,
    end=30.0,
):
    try:
        waveform, sample_rate = torchaudio.backend.soundfile_backend.load(file_path)
    except Exception as e:
        waveform, sample_rate = torchaudio.load(file_path)
    if waveform.shape[0] > 1:
        if is_mono:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if is_normalize:
        waveform = waveform / waveform.abs().max()
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
    
    # if sample_rate != target_sr:
    #     resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
    #     waveform = waveform.to(device)
    #     resampler = resampler.to(device)
    #     waveform = resampler(waveform)
    
    if return_start:
        return waveform, start
    path = os.path.join(cache_path, args.model+ "_" + "temp.wav")
    torchaudio.save(path, waveform, sample_rate)
    return path

for file_path in jsonl_list:
        work_list = [
            "ballroom_downbeat",
            "gtzan_beat",
            "ballroom_beat",
            "gtzan_downbeat",
            "Guzheng_Tech",
            "MedleyDB",
            "DSing"
            ]
        if not any([work in file_path for work in work_list]):
            continue
        results = []
        print(file_path)
        with open(file_path, "r") as file:
            lines = file.readlines()
        file.close
        
        count = 0
        for line in tqdm(lines):
            data = json.loads(line)
            if data['split'][0] != "test":
                continue
            count += 1
            if len(data['audio_path']) == 1: # input single audio
                prompt = data['instruction']
                audio_path = os.path.join(proj_path, data['audio_path'][0])
                label = data['output']
                start = data['audio_start']
                end = data['audio_end']
                temp_audio_path = split_and_save_audio(audio_path, start=start, end=end)
                try:
                    info, response = model.predict(temp_audio_path, prompt)
                except Exception as e:
                    print(e)
                    continue

                results.append( {
                    "question": prompt,
                    "response": response,
                    "correct_answer": label,
                    'audioid': audio_path,
                    'other': ""
                })

        if not os.path.exists(f"{args.output_file}/{args.model}"):
            os.mkdir(f"{args.output_file}/{args.model}")
        filename = f"{args.output_file}/{args.model}/{args.model}_{os.path.basename(file_path)[4:]}"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)

