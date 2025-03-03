import json
import tqdm
import os
import json
import argparse
import yaml
import glob

import torch
import torchaudio

from infer import (load_audio, 
                   get_qwen_pred, get_qwen2_pred, get_salmonn_pred, get_musilingo_pred, get_mullama_pred, 
                   prepare_audio_flamingo_tokenizer, prepare_audio_flamingo, get_audio_flamingo_pred,
                    get_ltu_pred
)
import re
import random
import numpy as np

# from answer_parsing import parse_multi_choice_response

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

path = "batch-5_1142_20240817.xlsx"
def split_string_by_options(input_string):
    pattern = r'(A\..*?)(B\..*?)(C\..*?)(D\..*)'
    matches = re.findall(pattern, input_string, re.DOTALL)
    return [match.strip() for match in matches[0]]

def get_jsonl_list(jsonl_rootpath):
    jsonl_list = [filepath for filepath in glob.glob(jsonl_rootpath + "/*/jsonl/understanding.jsonl") if "wav_xcodec_16k" not in filepath and "old" not in filepath]
    return jsonl_list

# get the jsonl filepath list
subsets = ["music", "audio", "speech"]
# all: understanding and generation
# jsonl_rootpaths = [f"/aifs4su/audio_flan/{subset}/jsonls_and_scp" for subset in subsets]
# understanding:
jsonl_rootpaths = [f"/aifs4su/mmcode/lmxue/dataset/audio_flan/{subset}_update_jsonl/split_generation_understanding/test/understanding" for subset in subsets]
file_list = [get_jsonl_list(jsonl_rootpath) for jsonl_rootpath in jsonl_rootpaths]
music_jsonl_list, audio_jsonl_list, speech_jsonl_list = [get_jsonl_list(jsonl_rootpath) for jsonl_rootpath in jsonl_rootpaths]
# jsonl_list = [
#     #         # # '18_NSynth/18_NSynth.jsonl', 
#     #     # # '18_NSynth/18_NSynth_CHOICE.jsonl', 
#     # '18_NSynth/NSynth_instrument_CHOICE.jsonl', 
#     # '18_NSynth/pitch_CHOICE.jsonl',
#     # #     # '18_NSynth/sft_data-18_NSynth-instrument_comparison.jsonl', 
#     # #     # '19_VocalSet_v1.2/19_VocalSet_v1.2.jsonl', 
#     # '42_GuitarSet/sft_data-42_GuitarSet.jsonl', 
#     # #     # '117_GTZAN-rhythm-Tempo_Comparison/sft_data-117_GTZAN-rhythm-Tempo_Comparison.jsonl', 
#     # #     # '119_GiantSteps-key-Key_Comparison/sft_data-119_GiantSteps-key-Key_Comparison.jsonl', 
#     # # # '128_Compmusic-Jingju_Audio_Recordings_Collection_english_version-music_captions/sft_data-128_Compmusic-Jingju_Audio_Recordings_Collection_english_version-music_captions.jsonl', 
#     # #     # '116_FreesoundLoopDataset/sft_data-116_FreesoundLoopDataset-tempo_comparison.jsonl', 
#     # '116_FreesoundLoopDataset/sft_data-116_FreesoundLoopDataset-genre-classification.jsonl', 
#     # '116_FreesoundLoopDataset/sft_data-116_FreesoundLoopDataset-key_detection.jsonl',
#     # '67_musicaps/sft_data-67_musicaps-music_captions.jsonl', 
#     # '155_vocadito/155_vocadito_PTR_sft.jsonl',
#     # #         # '66_MTG-Jamendo/66_MTG-Jamendo.jsonl'
#     # #         # '66_MTG-Jamendo/66_MTG-Jamendo_CHOICE.jsonl', 
#     # '50_AAM/sft_data-50_AAM-beat_tracking.jsonl', 
#     # '50_AAM/sft_data-50_AAM-chord_estimation.jsonl', 
#     # '50_AAM/sft_data-50_AAM-instrument_classification.jsonl', 
#     # '50_AAM/sft_data-50_AAM-instrument_recognition.jsonl', 
#     # '50_AAM/sft_data-50_AAM-key_detection.jsonl', 
#     # '50_AAM/sft_data-50_AAM-pitch_estimation.jsonl',
#     # "128_Compmusic-Jingju_Audio_Recordings_Collection_english_version-music_captions/sft_data-128_Compmusic-Jingju_Audio_Recordings_Collection_english_version-music_captions.jsonl",
#     # '66_MTG-Jamendo/emotion_CHOICE.jsonl', 
#     # '66_MTG-Jamendo/genre_CHOICE.jsonl', 
#     # '66_MTG-Jamendo/instrument_CHOICE.jsonl', 
#     '66_MTG-Jamendo/tag_CHOICE.jsonl',
#     ]

def transfer_qwen2_data_to_other_baselines(jsonl_filepath, baseline_model_name):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--output-file', default="/aifs4su/ziyaz/OmniBench/results", type=str, help='the path to save the output json file')
    
    parser.add_argument('--model', default="flamingo", type=str, 
                        choices=["qwen", "qwen2", "salmonn", "gpt-4o", "musilingo", "ltu", "ltuas", "mullama", "flamingo"], 
                        help='the model to use for inference')
    parser.add_argument("--cfg-path", type=str, default="/aifs4su/ziyaz/OmniBench/inference/audio_models/SALMONN/decode_config.yaml", help='path to configuration file')
    parser.add_argument(
        "--file-path", 
        type=str, 
        default="/data/ziyaz/audio_flan/Audio-Flan-Engine/datasets/music/116_FreesoundLoopDataset/0-genre_classification/sft_data-116_FreesoundLoopDataset-genre-classification.jsonl", 
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
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # default: all domains understanding.jsonl
    jsonl_list = music_jsonl_list + audio_jsonl_list + speech_jsonl_list
    if args.model == "qwen":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("/aifs4su/ziyaz/OmniBench/inference/audio_models/Qwen-Audio-Chat", trust_remote_code=True)
        qwen = AutoModelForCausalLM.from_pretrained("/aifs4su/ziyaz/OmniBench/inference/audio_models/Qwen-Audio-Chat", device_map="cuda", trust_remote_code=True).eval()
    elif args.model == "qwen2":
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        qwen2 = Qwen2AudioForConditionalGeneration.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", device_map="cuda", trust_remote_code=True).eval()
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
        musilingo = AutoModel.from_pretrained("/aifs4su/ziyaz/OmniBench/inference/audio_models/MusiLingo-long-v1", trust_remote_code=True)
        musilingo.to("cuda")
        musilingo.eval()
        jsonl_list = music_jsonl_list
        
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
        model_path = "/aifs4su/ziyaz/OmniBench/inference/audio_models/MU_LLaMA/ckpt/checkpoint.pth"
        llama_dir = "/aifs4su/ziyaz/OmniBench/inference/audio_models/MU_LLaMA/ckpt/LLaMA"
        model = llama.load(model_path, 
                           llama_dir, 
                           mert_path="/aifs4su/ziyaz/OmniBench/inference/audio_models/MERT-v1-330M", 
                           knn=True, 
                           knn_dir="/aifs4su/ziyaz/OmniBench/inference/audio_models/MU-LLaMA/ckpt", 
                           llama_type="7B")
        model.eval()
        model.to("cuda")
        jsonl_list = music_jsonl_list
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
        config_file = '/aifs4su/ziyaz/OmniBench/inference/audio_models/audio_flamingo/configs/foundation.yaml'
        config = yaml.load(open(config_file), Loader=yaml.FullLoader)
        clap_config = config['clap_config']
        model_config = config['model_config']
        text_tokenizer = prepare_audio_flamingo_tokenizer(model_config)
        audio_flamingo = prepare_audio_flamingo(
            model_config=model_config, 
            clap_config=clap_config, 
            checkpoint_path="/aifs4su/ziyaz/OmniBench/inference/audio_models/audio_flamingo/foundation.pt"
            # "/home/intern-2024-02/.cache/huggingface/hub/models--nvidia--audio-flamingo/snapshots/77da4908d6a7edae1f776810ebde708383eaedeb/foundation.pt"
        )
        audio_flamingo.eval()
        audio_flamingo.to("cuda")
    
    test_set = "qwen2-audio-subset"
    # file_path = args.file_path
    if test_set == "all":

        for file_path in jsonl_list:
            
            # file_path = os.path.join(
            #     "/data/ziyaz/OmniBench/music_local",
            #     file_path
            # )
            results = []
            with open(file_path, "r") as file:
                content = file.readlines()
                try:  # when json.dump(list of dict) is used
                    data_list = [json.loads(item) for item in content]  
                    # data_list = content
                except:  # when json.dump(dict + '\n') for each sample is used
                    data_list = [eval(tmp) for tmp in content.split("\n")[:-1]]
            print(data_list[0])
            #有人dataset["split"] 用 "test" 有人用 ["test"]
            if type(data_list[0]["split"]) == str: 
                data_list = [data for data in data_list if data["split"]== "test"]
            else:
                data_list = [data for data in data_list if data["split"][0] == "test"]
            
            # get the scp file path
            tmp = glob.glob(file_path.replace(".jsonl", "*.scp"))
            
            #有人dataset有多個scp
            if len(tmp) > 1: 
                tmp = [i for i in tmp if "audio" in i]
            tmp = tmp[0]
            with open(tmp, "r") as file:
                scp_content = file.read().split("\n")[:-1]
            audio_roots = {}

            for scp in scp_content:
                if len(scp.split("\t")) == 2:
                    key, path = scp.split("\t")
                else:
                    key, path = scp.split(" ") #有人scp 用空格而不是\t
                if key.startswith("42_GuitarSet__"): #Guitarset的scp前缀跟audio input不一样
                    key = re.sub( "42_GuitarSet__", "42_GuitarSet_original_datasets_",key)
                if key.endswith(".wav"): # key 作为文件名不包括 .wav
                    key = key.split(".")[0]
                if key.endswith(".aiff"): #FSLD 的 .aiff 有的是文件名写进scp 但没写进 data input, genre 和 key 处理不同
                    audio_roots[key] = path
                    key = key.split(".")[0]
                audio_roots[key] = path
            # print(key, path)

            # 没有写多个<|SOA|> <|EOA|>的情况，确认一下是否每个baselines都有，可能还是得单独写py文件定义jsonl_list
            for data in tqdm.tqdm(data_list):
                prompt = data["instruction"] + re.sub(r"<|SOA|>.*<|EOA|>", "", data["input"])
                try:
                    key = eval(data["input"].split("<|SOA|>")[1].split("<|EOA|>")[0]).split(".")[0]
                except:
                    key = data["input"].split("<|SOA|>")[1].split("<|EOA|>")[0]
                if key.endswith("wav") or key.endswith("mp3"):
                    key = key.split(".")[0]
                if "MTG" in key:
                    key = key.split("_")[-1] + ".low.mp3"

                audio_path = audio_roots[key]
                # print(key)
                # audio_path = re.sub("/scratch/buildlam/processed_data/codeclm/intermedia_jsonls/sft/datasets/",
                #                 "", 
                #                 audio_roots[key])
                # if "MTG" in audio_path:
                #     audio_path = re.sub('.low', '', audio_path)
                # transfer the old filepath into the new filepath
                # audio_path = re.sub('/scratch/buildlam/rawdata/codeclm/music//', 
                #                     "", audio_path)
                # audio_path = re.sub('/home/intern-2024-01/MARBLE-Benchmark/data/MTG/', 
                #                     "66_MTG-Jamendo/", audio_path)
                # audio_path = re.sub('/scratch/buildlam/rawdata/codeclm/music/original_datasets/', 
                #                     "", audio_path)
                # audio_path = re.sub('/scratch/buildlam/processed_data/codeclm/intermedia_jsonls/sft/original_datasets/',
                #                     "", audio_path)
                # audio_path = re.sub('/scratch/buildlam/rawdata/codeclm/music/vocadito_data/',
                #                     "155_vocadito/", audio_path)
                # audio_path = re.sub("/scratch/buildlam/rawdata/codeclm/music/50_AAM",
                #                     "50_AAM", audio_path)
                # audio_path = re.sub("/scratch/buildlam/rawdata/codeclm/music/128_Compmusic",
                #                     "128_Compmusic-Jingju_Audio_Recordings_Collection_english_version-music_captions/128_Compmusic", audio_path)
                # audio_path = os.path.join("/data/ziyaz/OmniBench/music_local/", audio_path)
                tmp = torchaudio.load(audio_path)
                # # break

                
                if args.model == "qwen":
                    response = get_qwen_pred(prompt, audio_path, tokenizer, qwen)
                elif args.model == "qwen2":
                    response = get_qwen2_pred(prompt, audio_path, processor, qwen2, sr)
                elif args.model == "salmonn":
                    response = get_salmonn_pred("<Speech><SpeechHere></Speech>" + prompt, audio_path, wav_processor, sal, cfg)
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
                    response = get_musilingo_pred(musilingo.model, prompt, audio_path, stopping, length_penalty=100, temperature=0.1)
                elif args.model == "ltu":
                    # # load audio from audio_path and resample to 16K and save again
                    # tmp = load_audio(audio_path, 16000, 
                    #                 is_normalize=False, crop_to_length_in_sample_points=16000*30+1, crop_randomly=True, pad=False)
                    # torchaudio.save("tmp_ltu_16kHz.wav", tmp, 16000)
                    # response = client.predict(
                    #     "tmp_ltu_16kHz.wav",
                    #     prompt, 
                    #     api_name="/predict"
                    # )
                    info, response = get_ltu_pred(ltu_model, tokenizer, prompter, prompt, audio_path)
                elif args.model == "ltuas":
                    tmp = load_audio(audio_path, 16000, 
                                    is_normalize=False, crop_to_length_in_sample_points=16000*30+1, crop_randomly=True, pad=False)
                    torchaudio.save("tmp_ltuas_16kHz.wav", tmp, 16000)
                    response = client.predict(
                                "tmp_ltuas_16kHz.wav",  # your audio file in 16K
                                "",
                                prompt,    # your question
                                "7B (Default)",    # str in 'LLM size' Radio component
                                api_name="/predict"
                    )
                elif args.model == "mullama":
                    response = get_mullama_pred(model, prompt, audio_path, 
                                                audio_weight=1, cache_size=100, cache_t=20.0, cache_weight=0.0, max_gen_len=1024, gen_t=0.6, top_p=0.8)
                elif args.model == "flamingo":
                    DataProcessor = AudioTextDataProcessor(
                            data_root="",
                            clap_config=clap_config,
                            tokenizer=text_tokenizer,
                            max_tokens=1024,
                            )
                    item = {'name': audio_path, 'prefix': 'The task is audio QA.', 'prompt': prompt}
                    processed_item = DataProcessor.process(item)
                    response = get_audio_flamingo_pred(
                        audio_flamingo, text_tokenizer, item, processed_item,
                        inference_kwargs,
                    )[0]

                results.append( {
                    "question": prompt,
                    "response": response,
                    "correct_answer": data["output"],
                    'audioid': audio_path,
                    'other': data["other"]
                })

            dataset_name = file_path.split("/")[-3]
            filename = f"{args.output_file}/{args.model}_{dataset_name}.jsonl"
            with open(filename, "w") as f:
                json.dump(results, f, indent=4)
    else:
        test_files = ["/aifs4su/ziyaz/ms-swift/datasets/audio_understanding_test_random_demo2.jsonl",
                      "/aifs4su/ziyaz/ms-swift/datasets/music_understanding_test_demo1.jsonl",
                      "/aifs4su/ziyaz/ms-swift/datasets/speech_understanding_test_random_demo2.jsonl"]
        for file_path in test_files:
            results = []
            with open(file_path, "r") as file:
                lines = file.readlines()
                for line in lines:
                    data = json.loads(line)
                    if len(data['audios']) == 1: # input single audio
                        prompt = data['query']
                        audio_path = data['audios'][0]
                        label = data['response']
                        if args.model == "qwen":
                            response = get_qwen_pred(prompt, audio_path, tokenizer, qwen)
                        elif args.model == "qwen2":
                            response = get_qwen2_pred(prompt, audio_path, processor, qwen2, sr)
                        elif args.model == "salmonn":
                            response = get_salmonn_pred("<Speech><SpeechHere></Speech>" + prompt, audio_path, wav_processor, sal, cfg)
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
                            response = get_musilingo_pred(musilingo.model, prompt, audio_path, stopping, length_penalty=100, temperature=0.1)
                        elif args.model == "ltu":
                            # # load audio from audio_path and resample to 16K and save again
                            # tmp = load_audio(audio_path, 16000, 
                            #                 is_normalize=False, crop_to_length_in_sample_points=16000*30+1, crop_randomly=True, pad=False)
                            # torchaudio.save("tmp_ltu_16kHz.wav", tmp, 16000)
                            # response = client.predict(
                            #     "tmp_ltu_16kHz.wav",
                            #     prompt, 
                            #     api_name="/predict"
                            # )
                            info, response = get_ltu_pred(ltu_model, tokenizer, prompter, prompt, audio_path)
                        elif args.model == "ltuas":
                            tmp = load_audio(audio_path, 16000, 
                                            is_normalize=False, crop_to_length_in_sample_points=16000*30+1, crop_randomly=True, pad=False)
                            torchaudio.save("tmp_ltuas_16kHz.wav", tmp, 16000)
                            response = client.predict(
                                        "tmp_ltuas_16kHz.wav",  # your audio file in 16K
                                        "",
                                        prompt,    # your question
                                        "7B (Default)",    # str in 'LLM size' Radio component
                                        api_name="/predict"
                            )
                        elif args.model == "mullama":
                            response = get_mullama_pred(model, prompt, audio_path, 
                                                        audio_weight=1, cache_size=100, cache_t=20.0, cache_weight=0.0, max_gen_len=1024, gen_t=0.6, top_p=0.8)
                        elif args.model == "flamingo":
                            DataProcessor = AudioTextDataProcessor(
                                    data_root="",
                                    clap_config=clap_config,
                                    tokenizer=text_tokenizer,
                                    max_tokens=1024,
                                    )
                            item = {'name': audio_path, 'prefix': 'The task is audio QA.', 'prompt': prompt}
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

        dataset_name = file_path.split("/")[-3]
        filename = f"{args.output_file}/{args.model}_{dataset_name}.jsonl"
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)
    
