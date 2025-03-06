import os
import torch
import torchaudio
from peft import (
    LoraConfig,
    get_peft_model,
)
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import datetime
import time,json
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
    
def load_audio(audio_path):
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

def predict(self, audio_path, question):
    print('audio path, ', audio_path)
    begin_time = time.time()

    instruction = question
    prompt = prompter.generate_prompt(instruction, None)
    print('Input prompt: ', prompt)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)

    if audio_path != None:
        cur_audio_input, audio_info = load_audio(audio_path)
        cur_audio_input = cur_audio_input.unsqueeze(0)
        if torch.cuda.is_available() == False:
            pass
        else:
            cur_audio_input = cur_audio_input.half().to(device)
    else:
        print('go to the none audio loop')
        cur_audio_input = None
        audio_info = 'Audio is not provided, answer pure language question.'

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
            input_ids=input_ids.to(device),
            audio_input=cur_audio_input,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=400,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)[len(prompt)+6:-4] # trim <s> and </s>
    end_time = time.time()
    print(output)
    cur_res = {'audio_id': audio_path, 'input': instruction, 'output': output}
    eval_log.append(cur_res)
    with open(log_save_path, 'w') as outfile:
        json.dump(eval_log, outfile, indent=1)
    print('eclipse time: ', end_time - begin_time, ' seconds.')
    return audio_info, output

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # no matter which check point you use, do not change this section, this loads the llm
    prompter = Prompter('alpaca_short')
    tokenizer = LlamaTokenizer.from_pretrained('../../pretrained_mdls/vicuna_ltu/')
    if device == 'cuda':
        model = LlamaForCausalLM.from_pretrained('../../pretrained_mdls/vicuna_ltu/', device_map="auto", torch_dtype=torch.float16)
    else:
        model = LlamaForCausalLM.from_pretrained('../../pretrained_mdls/vicuna_ltu/', device_map="auto")

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)

    # change the path to your checkpoint
    state_dict = torch.load('/data/siyou/CMI-bench/pretrained_models/LTU/ltu_ori_paper.bin', map_location='cpu')
    msg = model.load_state_dict(state_dict, strict=False)

    model.is_parallelizable = True
    model.model_parallel = True

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

    SAMPLE_RATE = 16000
    AUDIO_LEN = 1.0
    sample_audio_text = "[sample audios from AudioSet evaluation set]"

