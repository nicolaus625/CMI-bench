import sys
sys.path.append('/import/c4dm-04/siyoul/CMI-bench/eval/Pengi')
from wrapper import PengiWrapper as Pengi
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class PengiModel:
    def __init__(self, device="cuda:0"):
        self.pengi = Pengi(config="base")
        self.device = device


    def predict(self, audio_paths, prompt, max_len=30, beam_size=3, temperature=1.0, stop_token=' <|endoftext|>'):
        
        generated_summary = self.pengi.describe(audio_paths=[audio_paths],
                                            max_len=max_len, 
                                            beam_size=beam_size, 
                                            temperature=temperature, 
                                            stop_token=stop_token
                                            )
        return None, generated_summary[0]


if __name__ == "__main__":
    device = "cuda:0"
    model = PengiModel()
    print(model.predict("/import/c4dm-04/siyoul/CMI-bench/res/example/f2_arpeggios_belt_a_00.wav"))