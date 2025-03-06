import json
import os
import torch
import time
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import numpy as np
import datetime
import re
import skimage.measure
import whisper_at
from whisper.model import Whisper, ModelDimensions
import os.path as osp
from typing import Union

templates_path = "/data/siyou/CMI-bench/res/templates"

class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join(templates_path, f"{template_name}.json")
        if not osp.exists(file_name):
            file_name = osp.join(templates_path, f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()

def convert_params_to_float32(model):
    for name, param in model.named_parameters():
        if "audio_encoder" in name and "ln" in name:
            if param.dtype == torch.float16:
                print(f"Converting parameter '{name}' to float32")
                param.data = param.data.float()



def load_whisper():
    mdl_size = 'large-v1'
    checkpoint_path = '../../pretrained_mdls/{:s}.pt'.format(mdl_size)
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    dims = ModelDimensions(**checkpoint["dims"])
    whisper_feat_model = Whisper(dims)
    whisper_feat_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    whisper_feat_model.to('cuda:0')
    return whisper_feat_model

class LTUASModel:
    def __init__(self, base_model, eval_mdl_path, device):
        self.prompter = Prompter('alpaca_short')
        self.tokenizer = LlamaTokenizer.from_pretrained(base_model)
        if device == 'cuda':
            self.model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
        else:
            self.model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto")
        convert_params_to_float32(self.model)

        config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(self.model, config)

        temp, top_p, top_k = 0.1, 0.95, 500

        state_dict = torch.load(eval_mdl_path, map_location='cpu')
        miss, unexpect = self.model.load_state_dict(state_dict, strict=False)

        self.model.is_parallelizable = True
        self.model.model_parallel = True

        # unwind
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_text_model = whisper_at.load_model("large-v2", device='cuda:1')
whisper_feat_model = load_whisper()



tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == 'cuda':
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16)
else:
    model = LlamaForCausalLM.from_pretrained(base_model, device_map="auto")
convert_params_to_float32(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

temp, top_p, top_k = 0.1, 0.95, 500

state_dict = torch.load(eval_mdl_path, map_location='cpu')
miss, unexpect = model.load_state_dict(state_dict, strict=False)

model.is_parallelizable = True
model.model_parallel = True

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

eval_log = []
cur_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_save_path = './inference_log/'
if os.path.exists(log_save_path) == False:
    os.mkdir(log_save_path)
log_save_path = log_save_path + cur_time + '.json'

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

text_cache = {}
def load_audio_trans(filename):
    global text_cache
    if filename not in text_cache:
        result = whisper_text_model.transcribe(filename)
        text = remove_thanks_for_watching(result["text"].lstrip())
        text_cache[filename] = text
    else:
        text = text_cache[filename]
        print('using asr cache')
    _, audio_feat = whisper_feat_model.transcribe_audio(filename)
    audio_feat = audio_feat[0]
    audio_feat = torch.permute(audio_feat, (2, 0, 1)).detach().cpu().numpy()
    audio_feat = skimage.measure.block_reduce(audio_feat, (1, 20, 1), np.mean)
    audio_feat = audio_feat[1:]  # skip the first layer
    audio_feat = torch.FloatTensor(audio_feat)
    return audio_feat, text

# trim to only keep output
def trim_string(a):
    separator = "### Response:\n"
    trimmed_string = a.partition(separator)[-1]
    trimmed_string = trimmed_string.strip()
    return trimmed_string

def predict(audio_path, question):
    print('audio path, ', audio_path)
    begin_time = time.time()

    if audio_path != None:
        cur_audio_input, cur_input = load_audio_trans(audio_path)
        if torch.cuda.is_available() == False:
            pass
        else:
            cur_audio_input = cur_audio_input.unsqueeze(0).half().to(device)

    instruction = question
    prompt = prompter.generate_prompt(instruction, cur_input)
    print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

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
    end_time = time.time()
    print(trim_string(output))
    cur_res = {'audio_id': audio_path, 'instruction': instruction, 'input': cur_input, 'output': trim_string(output)}
    eval_log.append(cur_res)
    with open(log_save_path, 'w') as outfile:
        json.dump(eval_log, outfile, indent=1)
    print('eclipse time: ', end_time-begin_time, ' seconds.')
    return trim_string(output)

if __name__ == '__main__':
    # do not change this, this will load llm
    base_model = "../../pretrained_mdls/vicuna_ltuas/"
    # change this to your checkpoint
    eval_mdl_path = '../../pretrained_mdls/ltuas_long_noqa_a6.bin'
    eval_mode = 'joint'