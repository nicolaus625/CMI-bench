import json
from tqdm import tqdm
import os
import json
import argparse
import yaml
import glob

import torch
import torchaudio

import re
import random
import numpy as np

from data_loader import load_audio


HF_PATH = "/map-vepfs/yinghao/huggingface"

def parse_multi_choice_response(response, all_choices, index2ans, default_answer=None):
    """
    Parse the prediction from the generated response.
    Return the predicted index e.g., A, B, C, D.
    https://github.com/MMMU-Benchmark/MMMU/blob/51ce7f3e829c16bb44bc5445782686b4c3508794/eval/eval_utils.py#L10
    """
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = " " + response + " "  # add space to avoid partial match

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:  # e.g., (A) (B) (C) (D)
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A B C D
            if f"{choice} " in response:
                candidates.append(choice)

    if len(candidates) == 0:
        for choice in all_choices:  # e.g., A. B. C. D.
            if f"{choice}." in response:
                candidates.append(choice)

    # if all above doesn't get candidates, check if the content is larger than 5 tokens and try to parse the example
    if len(candidates) == 0 and len(response.split()) > 5:
        for index, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(index)
                index_ans = False  # it's content ans.

    if len(candidates) == 0:  # still not get answer, randomly choose one.
        if default_answer is None:
            pred_index = random.choice(all_choices)
        else:
            pred_index = default_answer
    elif len(candidates) > 1:
        start_indexes = []
        if index_ans:
            if ans_with_brack:
                for can in candidates:
                    index = response.rfind(f"({can})")
                    start_indexes.append(index)  # -1 will be ignored anyway
                # start_indexes = [generated_response.index(f'({can})') for can in candidates]
            else:
                for can in candidates:
                    index = response.rfind(f" {can} ")
                    start_indexes.append(index)
        else:
            for can in candidates:
                index = response.lower().rfind(index2ans[can].lower())
                start_indexes.append(index)
        # get the last one
        pred_index = candidates[np.argmax(start_indexes)]
    else:  # if only one candidate, use it.
        pred_index = candidates[0]

    return pred_index

def split_string_by_options(input_string):
    pattern = r'(A\..*?)(B\..*?)(C\..*?)(D\..*)'
    matches = re.findall(pattern, input_string, re.DOTALL)
    return [match.strip() for match in matches[0]]


def transfer_qwen2_data_to_other_baselines(jsonl_filepath, baseline_model_name):
    return

def get_qwen_pred(text, audio_path, tokenizer, model):
    query = tokenizer.from_list_format([
        {'audio': audio_path}, # Either a local path or an url
        {'text': text},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    return response        

def get_qwen2_pred(text, audio_path, processor, model, sr):
    conversation = [{'role': 'system', 'content': 'You are a helpful assistant.'}, 
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": text},
        ]}
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios = []
    for message in conversation:
        if isinstance(message["content"], list):
            for ele in message["content"]:
                if ele["type"] == "audio":
                    audio = load_audio(
                            ele["audio"],
                            target_sr=sr,
                            is_mono=True,
                            is_normalize=False,
                            pad=False,
                        )
                    audio = audio.squeeze(0).numpy()
                    audios.append(audio)
    inputs = processor(text=text, audios=audios, return_tensors="pt", padding=True).to("cuda")
    inputs.input_ids = inputs.input_ids.to("cuda")

    generate_ids = model.generate(**inputs, max_length=1024)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]

    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response


def get_salmonn_pred(text, audio_path, processor, model, cfg):
    audio = load_audio(audio_path, target_sr=16000,
                        is_mono=True,
                        is_normalize=False,
                        pad=False).squeeze(0).numpy().astype(np.float64)
    # import soundfile as sf

    # audio, sr = sf.read(audio_path)
    # if len(audio.shape) == 2: # stereo to mono
    #     audio = audio[:, 0]
    # if len(audio) < sr: # pad audio to at least 1s
    #     sil = np.zeros(sr - len(audio), dtype=float)
    #     audio = np.concatenate((audio, sil), axis=0)
    # audio = audio[: sr * 30] # truncate audio to at most 30s
    spectrogram = processor(audio, sampling_rate=16000, return_tensors="pt")["input_features"]
    samples = {
        "spectrogram": spectrogram.cuda(),
        "raw_wav": torch.from_numpy(audio).unsqueeze(0).cuda(),
        "padding_mask": torch.zeros(len(audio), dtype=torch.bool).unsqueeze(0).cuda(),
    }
    prompt = [text]
    # for environment with cuda>=117
    with torch.cuda.amp.autocast(dtype=torch.float16):
        response = model.generate(samples, cfg.config.generate, prompts=prompt)[0]
    return response


def get_musilingo_pred(model, text, audio_path, stopping, length_penalty=1, temperature=0.1):
    max_new_tokens=300
    num_beams=1
    min_length=1
    top_p=0.5
    repetition_penalty=1.0
    from transformers import Wav2Vec2FeatureExtractor
    audio = load_audio(audio_path, target_sr=24000,
                        is_mono=True,
                        is_normalize=False,
                        pad=False).cuda()    
    processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-330M",trust_remote_code=True) 
    audio = processor(audio, 
                    sampling_rate=24000, 
                    return_tensors="pt")['input_values'][0].cuda() 
        
    audio_embeds, atts_audio = model.encode_audio(audio)
        
    prompt = '<Audio><AudioHere></Audio> ' + text
    instruction_prompt = [model.prompt_template.format(prompt)]
    audio_embeds, atts_audio = model.instruction_prompt_wrap(audio_embeds, atts_audio, instruction_prompt)
    
    model.llama_tokenizer.padding_side = "right"
    batch_size = audio_embeds.shape[0]
    bos = torch.ones([batch_size, 1],
                    dtype=torch.long,
                    device=torch.device('cuda')) * model.llama_tokenizer.bos_token_id
    bos_embeds = model.llama_model.model.embed_tokens(bos)
    # atts_bos = atts_audio[:, :1]
    inputs_embeds = torch.cat([bos_embeds, audio_embeds], dim=1)
    # attention_mask = torch.cat([atts_bos, atts_audio], dim=1)
    outputs = model.llama_model.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping,
        num_beams=num_beams,
        do_sample=True,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
    )
    output_token = outputs[0]
    if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
        output_token = output_token[1:]
    if output_token[0] == 1:  # if there is a start token <s> at the beginning. remove it
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('###')[0]  # remove the stop sign '###'
    output_text = output_text.split('Assistant:')[-1].strip()
    return output_text


def get_mullama_pred(
        model,
        prompt,
        audio_path,
        audio_weight=1,
        cache_size=100,
        cache_t=20.0,
        cache_weight=0.0,
        max_gen_len=1024,
        gen_t=0.6, 
        top_p=0.8
):
    inputs = {}
    audio = load_and_transform_audio_data([audio_path])
    inputs['Audio'] = [audio, audio_weight]
    image_prompt = prompt
    text_output = None
    prompts = [llama.format_prompt(prompt)]
    prompts = [model.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
    with torch.cuda.amp.autocast():
        results = model.generate(inputs, prompts, max_gen_len=max_gen_len, temperature=gen_t, top_p=top_p,
                                     cache_size=cache_size, cache_t=cache_t, cache_weight=cache_weight)
    text_output = results[0].strip()
    return text_output


def prepare_audio_flamingo_tokenizer(model_config):
    from transformers import AutoTokenizer
    tokenizer_path = model_config['tokenizer_path']
    cache_dir = model_config['cache_dir']
    text_tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=False,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<audio>", "<|endofchunk|>"]}
    ) 
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if text_tokenizer.sep_token is None:
        text_tokenizer.add_special_tokens({"sep_token": "<SEP>"})
    return text_tokenizer

def prepare_audio_flamingo(model_config, clap_config, checkpoint_path, device_id=0):
    from audio_flamingo.src.factory import create_model_and_transforms
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # disable the tokenizer parallelism warning
    model, tokenizer = create_model_and_transforms(
        **model_config,
        clap_config=clap_config,
        use_local_files=False,
        gradient_checkpointing=False,
        freeze_lm_embeddings=False,
    )
    model.eval()
    model = model.to(device_id)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = checkpoint["model_state_dict"]
    model_state_dict = {k.replace("module.", ""): v for k, v in model_state_dict.items()}
    model.load_state_dict(model_state_dict, False)

    return model


def get_audio_flamingo_pred(model, tokenizer, item, processed_item, inference_kwargs, device_id=0):
    filename, audio_clips, audio_embed_mask, input_ids, attention_mask = processed_item
    audio_clips = audio_clips.to(device_id, dtype=None, non_blocking=True)
    audio_embed_mask = audio_embed_mask.to(device_id, dtype=None, non_blocking=True)
    input_ids = input_ids.to(device_id, dtype=None, non_blocking=True).squeeze()

    media_token_id = tokenizer.encode("<audio>")[-1]
    eoc_token_id = tokenizer.encode("<|endofchunk|>")[-1]
    sep_token_id = tokenizer.sep_token_id
    eos_token_id = tokenizer.eos_token_id
    
    outputs = model.generate(
        audio_x=audio_clips.unsqueeze(0),
        audio_x_mask=audio_embed_mask.unsqueeze(0),
        lang_x=input_ids.unsqueeze(0),
        eos_token_id=eos_token_id,
        max_new_tokens=128,
        **inference_kwargs,
    )

    outputs_decoded = [
        tokenizer.decode(output).split(tokenizer.sep_token)[-1].replace(tokenizer.eos_token, '').replace(tokenizer.pad_token, '').replace('<|endofchunk|>', '') for output in outputs
    ]

    return outputs_decoded


def get_ltu_pred(model, tokenizer, prompter, text, audio_path):
    from transformers import GenerationConfig
    
    def load_ltu_audio(audio_path):
        # print('audio_path: ', audio_path)
        waveform, sample_rate = torchaudio.load(audio_path)
        audio_info = 'Original input audio length {:.2f} seconds, number of channels: {:d}, sampling rate: {:d}.'.format(waveform.shape[1]/sample_rate, waveform.shape[0], sample_rate)
        if waveform.shape[0] != 1:
            waveform = waveform[0].unsqueeze(0)
            audio_info += ' Only the first channel is used.'
        if sample_rate == 16000:
            pass
        else:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)
            sample_rate = 16000
            audio_info += ' Resample to 16000Hz.'
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sample_rate,
                                                use_energy=False, window_type='hanning',
                                                num_mel_bins=128, dither=0.0, frame_shift=10)
        target_length = 1024
        n_frames = fbank.shape[0]
        p = target_length - n_frames
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]
        # normalize the fbank
        fbank = (fbank + 5.081) / 4.4849
        return fbank, audio_info

    prompt = prompter.generate_prompt(text, None)
    # print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    cur_audio_input, audio_info = load_ltu_audio(audio_path)
    cur_audio_input = cur_audio_input.unsqueeze(0)
    cur_audio_input = cur_audio_input.half().to("cuda")
    
    generation_config = GenerationConfig(
        do_sample=True,
        temperature=0.1,
        top_p=0.95,
        max_new_tokens=400,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids.to("cuda"),
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=400,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)[len(prompt)+6:-4] # trim <s> and </s>
    return audio_info, output


def convert_params_to_float32(model):
    for name, param in model.named_parameters():
        if "audio_encoder" in name and "ln" in name:
            if param.dtype == torch.float16:
                print(f"Converting parameter '{name}' to float32")
                param.data = param.data.float()

def print_parameters(model):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, Data type: {param.dtype}, device '{param.device}'")

def remove_thanks_for_watching(text):
    variations = [
        "thanks for watching", "Thanks for watching", "THANKS FOR WATCHING",
        "thanks for watching.", "Thanks for watching.", "THANKS FOR WATCHING.",
        "thanks for watching!", "Thanks for watching!", "THANKS FOR WATCHING!",
        "thank you for watching", "Thank you for watching", "THANK YOU FOR WATCHING",
        "thank you for watching.", "Thank you for watching.", "THANK YOU FOR WATCHING.",
        "thank you for watching!", "Thank you for watching!", "THANK YOU FOR WATCHING!"
    ]
    variations = sorted(variations, key=len, reverse=True)
    pattern = "|".join(re.escape(var) for var in variations)
    result = re.sub(pattern, "", text)
    return result

# trim to only keep output
def trim_string(a):
    separator = "### Response:\n"
    trimmed_string = a.partition(separator)[-1]
    trimmed_string = trimmed_string.strip()
    return trimmed_string

def get_ltuas_predict(model, tokenizer, text, audio_path):
    def load_audio_trans(filename):
        global text_cache
        result = whisper_text_model.transcribe(filename)
        text = remove_thanks_for_watching(result["text"].lstrip())
        _, audio_feat = whisper_feat_model.transcribe_audio(filename)
        audio_feat = audio_feat[0]
        audio_feat = torch.permute(audio_feat, (2, 0, 1)).detach().cpu().numpy()
        audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
        audio_feat = audio_feat[1:]  # skip the first layer
        audio_feat = torch.FloatTensor(audio_feat)
        return audio_feat, text

    if audio_path != None:
        cur_audio_input, cur_input = load_audio_trans(audio_path)
        cur_audio_input = cur_audio_input.unsqueeze(0).half().to("cuda")

    prompt = prompter.generate_prompt(text, cur_input)
    # print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to("cuda")

    generation_config = GenerationConfig(
        do_sample=True,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=1.1,
        max_new_tokens=500,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        num_return_sequences=1
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=500,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    output = output[5:-4]
    return trim_string(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-file', default="results", type=str, help='the path to save the output json file')
    
    parser.add_argument('--model', default="qwen2", type=str, 
                        choices=["qwen", "qwen2", "salmonn", "gpt-4o", "musilingo", "ltu", "ltuas", "mullama", "flamingo"], 
                        help='the model to use for inference')
    parser.add_argument("--cfg-path", type=str, default="SALMONN/decode_config.yaml", help='path to configuration file')
    parser.add_argument(
        "--file-path", 
        type=str, 
        default="/map-vepfs/yinghao/CMI-bench/data/Beat-Transformer/CMI_ballroom_beat.jsonl", 
        help='path to jsonl'
    )
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    os.makedirs(args.output_file, exist_ok=True)

    # default: all domains understanding.jsonl
    jsonl_list = glob.glob("../data/*/CMI*.jsonl")
    if args.model == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(f"{HF_PATH}/Qwen-Audio-Chat", trust_remote_code=True)
        qwen = AutoModelForCausalLM.from_pretrained(f"{HF_PATH}/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
    elif args.model == "qwen2":
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained(f"{HF_PATH}/Qwen2-Audio-7B-Instruct")
        qwen2 = Qwen2AudioForConditionalGeneration.from_pretrained(f"{HF_PATH}/Qwen2-Audio-7B-Instruct", device_map="cuda", trust_remote_code=True).eval()
        sr = processor.feature_extractor.sampling_rate

    elif args.model == "salmonn":
        from transformers import WhisperFeatureExtractor
        from SALMONN.config import Config
        from SALMONN.models.salmonn import SALMONN
        print(args, args.cfg_path)
        cfg = Config(args)
        sal = SALMONN.from_config(cfg.config.model)
        sal.to("cuda")
        sal.eval()
        wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    elif args.model == "gpt-4o":
        NotImplementedError

    elif args.model == "musilingo":
        from transformers import Wav2Vec2FeatureExtractor
        from transformers import StoppingCriteria, StoppingCriteriaList
        from transformers import AutoModel
        musilingo = AutoModel.from_pretrained(f"{HF_PATH}/MusiLingo-long-v1", trust_remote_code=True)
        musilingo.to("cuda")
        musilingo.eval()
        
    elif args.model == "ltu":
        # from gradio_client import Client
        # client = Client("https://yuangongfdu-ltu.hf.space/")
        from peft import LoraConfig, get_peft_model
        from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
        from ltu.src.ltu.utils.prompter import Prompter

        prompter = Prompter('alpaca_short')
        tokenizer = LlamaTokenizer.from_pretrained('/aifs4su/ziyaz/OmniBench/inference/audio_models/ltu/pretrained_mdls/vicuna_ltu/')
        ltu_model = LlamaForCausalLM.from_pretrained('/aifs4su/ziyaz/OmniBench/inference/audio_models/ltu/pretrained_mdls/vicuna_ltu/', device_map="auto", torch_dtype=torch.float16)
        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        ltu_model = get_peft_model(ltu_model, config)
        state_dict = torch.load('/aifs4su/ziyaz/OmniBench/inference/audio_models/ltu/pretrained_mdls/ltu_ori_paper.bin', 
                                map_location='cpu')
        msg = ltu_model.load_state_dict(state_dict, strict=False)
        ltu_model.is_parallelizable = True
        ltu_model.model_parallel = True
        ltu_model.config.pad_token_id = tokenizer.pad_token_id = 0
        ltu_model.config.bos_token_id = 1
        ltu_model.config.eos_token_id = 2
        ltu_model.eval()
        jsonl_list = speech_jsonl_list + audio_jsonl_list
    elif args.model == "ltuas":
        from gradio_client import Client
        client = Client("https://yuangongfdu-ltu-2.hf.space/")
        jsonl_list = speech_jsonl_list + audio_jsonl_list
    elif args.model == "mullama":
        from MU_LLaMA.MU_LLaMA import llama
        from MU_LLaMA.MU_LLaMA.util.misc import *
        from MU_LLaMA.MU_LLaMA.data.utils import load_and_transform_audio_data
        model_path = "MU_LLaMA/ckpt/checkpoint.pth"
        llama_dir = "MU_LLaMA/ckpt/LLaMA"
        model = llama.load(model_path, 
                           llama_dir, 
                           mert_path=f"{HF_PATH}/MERT-v1-330M", 
                           knn=True, 
                           knn_dir="MU-LLaMA/ckpt", 
                           llama_type="7B")
        model.eval()
        model.to("cuda")
    elif args.model == "flamingo":
        from transformers import AutoTokenizer, set_seed 
        set_seed(0)
        from audio_flamingo.data import AudioTextDataProcessor
        from audio_flamingo.src.factory import create_model_and_transforms
        inference_kwargs = {
            "do_sample": True,
            "top_k": 30,
            "top_p": 0.95,
            "num_return_sequences": 1
        }
        config_file = 'audio_flamingo/configs/foundation.yaml'
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        clap_config = config['clap_config']
        model_config = config['model_config']
        text_tokenizer = prepare_audio_flamingo_tokenizer(model_config)
        audio_flamingo = prepare_audio_flamingo(
            model_config=model_config, 
            clap_config=clap_config, 
            checkpoint_path="audio_flamingo/foundation.pt"
            # "/home/intern-2024-02/.cache/huggingface/hub/models--nvidia--audio-flamingo/snapshots/77da4908d6a7edae1f776810ebde708383eaedeb/foundation.pt"
        )
        audio_flamingo.eval()
        audio_flamingo.to("cuda")
    
    for file_path in jsonl_list:
        # if "MTG" not in file_path:
        #     continue
        results = []
        with open(file_path, "r") as file:
            lines = file.readlines()
        file.close
        
        count = 0
        for line in tqdm(lines):
            data = json.loads(line)
            if data['split'][0] != "test" or count > 4:
                continue
            count += 1
            if len(data['audio_path']) == 1: # input single audio
                prompt = data['instruction']
                audio_path = "../test" + data['audio_path'][0]
                label = data['output']
                start = data['audio_start']
                end = data['audio_end']
                if args.model == "qwen":
                    # print(start, end, audio_path)
                    tmp = load_audio(audio_path, target_sr=16000, start=start, end=end)
                    torchaudio.save("tmp.wav", tmp, 16000)
                    response = get_qwen_pred(prompt, "tmp.wav", tokenizer, qwen)
                elif args.model == "qwen2":
                    tmp = load_audio(audio_path, target_sr=16000, start=start, end=end)
                    torchaudio.save("tmp.wav", tmp, 16000)
                    response = get_qwen2_pred(prompt, "tmp.wav", processor, qwen2, sr)
                elif args.model == "salmonn":
                    tmp = load_audio(audio_path, target_sr=16000, start=start, end=end)
                    torchaudio.save("tmp.wav", tmp, 16000)
                    response = get_salmonn_pred(f"USER: <Speech><SpeechHere></Speech>{prompt.strip()}\nASSISTANT:",
                                                "tmp.wav", wav_processor, sal, cfg)
                elif args.model == "gpt-4o":
                    NotImplementedError
                elif args.model == "musilingo":
                    class StoppingCriteriaSub(StoppingCriteria):
                        def __init__(self, stops=[], encounters=1):
                            super().__init__()
                            self.stops = stops
                        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                            for stop in self.stops:
                                if torch.all((stop == input_ids[0][-len(stop):])).item():
                                    return True
                            return False
                    stopping = StoppingCriteriaList([StoppingCriteriaSub([torch.tensor([835]).cuda(),
                                                    torch.tensor([2277, 29937]).cuda()])])
                    tmp = load_audio(audio_path, target_sr=24000, start=start, end=end)
                    torchaudio.save("tmp.wav", tmp, 24000)
                    response = get_musilingo_pred(musilingo.model, prompt, "tmp.wav", stopping, length_penalty=100, temperature=0.1)
                elif args.model == "ltu":
                    NotImplementedError
                elif args.model == "ltuas":
                    NotImplementedError
                elif args.model == "mullama":
                    tmp = load_audio(audio_path, target_sr=24000, start=start, end=end)
                    torchaudio.save("tmp.wav", tmp, 24000)
                    response = get_mullama_pred(model, prompt, "tmp.wav", 
                                                audio_weight=1, cache_size=100, cache_t=20.0, cache_weight=0.0, max_gen_len=1024, gen_t=0.6, top_p=0.8)
                elif args.model == "flamingo":
                    DataProcessor = AudioTextDataProcessor(
                            data_root="",
                            clap_config=clap_config,
                            tokenizer=text_tokenizer,
                            max_tokens=1024,
                            )
                    tmp = load_audio(audio_path, target_sr=44100, start=start, end=end)
                    torchaudio.save("tmp.wav", tmp, 44100)
                    item = {'name': "tmp.wav", 'prefix': 'The task is audio QA.', 'prompt': prompt}
                    processed_item = DataProcessor.process(item)
                    response = get_audio_flamingo_pred(
                        audio_flamingo, text_tokenizer, item, processed_item,
                        inference_kwargs,
                    )[0]

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
    
