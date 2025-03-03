# audio-model evaluation
## qwen-audio
pip install -r requirements.txt

inference with image caption and audio
> CUDA_VISIBLE_DEVICES=0 python inference/audio_models/infer.py --model qwen 
inference only with audio
> CUDA_VISIBLE_DEVICES=0 python inference/audio_models/infer.py --model qwen --no-image
inference only with image caption
> CUDA_VISIBLE_DEVICES=0 python inference/audio_models/infer.py --model qwen --no-audio

## qwen2-audio
pip install git+https://github.com/huggingface/transformers

this environment is conpatable with qwen-audio

inference with image caption and audio
> CUDA_VISIBLE_DEVICES=0 python inference/audio_models/infer.py --model qwen2 
inference only with audio
> CUDA_VISIBLE_DEVICES=0 python inference/audio_models/infer.py --model qwen2 --no-image
inference only with image caption
> CUDA_VISIBLE_DEVICES=0 python inference/audio_models/infer.py --model qwen2 --no-audio

## salmonn
### setup environment
pip install -r requirements_salmonn.txt

download beats ckpt from onedrive: https://onedrive.live.com/?authkey=%21APLo1x9WFLcaKBI&id=6B83B49411CA81A7%2125955&cid=6B83B49411CA81A7&parId=root&parQt=sharedby&o=OneUp

if you cannot download this link, use wget https://djf19a.dm.files.1drv.com/y4mwa3IjEyKMl5kK57BQ65XVHjIuMPD428GnN1PY9qkBfoWBAT1PURVSnl7Mq5fIRM00MWe0_IwM8F-PjWimiXCiCs_sZf6jy3-LaJtsnKzfTvQ5ogwyJvEnOdhWBnAizxkLS5EzrutX2U6GbQaMejUK6CN8DcJsjPXGGbBZfTWZN1kR7icLDiaujL5_4zrb0CBAZc30DPXWNmYsiTNqVHfKA and rename the file to BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2

download salmonn ckpt: wget https://huggingface.co/tsinghua-ee/SALMONN/resolve/main/salmonn_v1.pth

### inference commands
inference with image caption and audio
> CUDA_VISIBLE_DEVICES=1 python inference/audio_models/infer.py --model salmonn 
inference only with audio
> CUDA_VISIBLE_DEVICES=1 python inference/audio_models/infer.py --model salmonn --no-image
inference only with image caption
> CUDA_VISIBLE_DEVICES=1 python inference/audio_models/infer.py --model salmonn --no-audio

## audio flamingo
download clapcap_weights_2023.pth from for ms_clap https://huggingface.co/microsoft/msclap/tree/main

download 630k-fusion-best.pt from for laion_clap https://huggingface.co/lukewys/laion_clap/tree/main 

install the audio flamingo requirement with transformers==4.27.4

(recommended) conda env create -f environment.yaml

pip install -r requirements_audio_flamingo.txt

inference with image caption and audio
> CUDA_VISIBLE_DEVICES=1 python inference/audio_models/infer.py --model flamingo 
inference only with audio

## musilingo
use the same environment with salmon, but run the following command
> pip uninstall torchaudio
> pip install timm
> pip install torchaudio
> pip install omegaconf==2.3.0
> pip install nnAudio==0.3.3

inference with image caption and audio
> CUDA_VISIBLE_DEVICES=1 python inference/audio_models/infer.py --model musilingo 

## LTU and LTU-AS
use the same environment of qwen-audio or run the following command
> pip install gradio_client

## Mu-llama
> pip install -r requirements_mullama.txt
download the ckpt to ```inference/audio_models/MU_LLaMA/ckpt```. See more information at ```inference/audio_models/MU-LLaMA/README.md```.

inference with audio and image caption
> CUDA_VISIBLE_DEVICES=1 python inference/audio_models/infer.py --model mullama 

# use audio content annotation and a 0.5 second blank audio to replace audio itself
run ```text_infer.py``` instead of ```infer.py``` in the previous codes