import glob

import argparse
import json
import os
import torch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="qwen2", type=str, 
                        choices=["qwen", "qwen2", "salmonn", "gpt-4o", "musilingo", "ltu", "ltuas", "mullama", "flamingo"], 
                        help='the model to use for inference')
    
    args = parser.parse_args()
    model = args.model 
    results_json = glob.glob(f"model/results_test/{model}/{model}*.jsonl")
    result = results_json[0]
    task = os.path.basename(result)[len(model)+1:-6]
    # load jsonl
    with open(result, "r") as f:
        # data = [json.load(line.strip()) for line in f]
        data = json.load(f)
    f.close()
    
    for sample in data:
        print(sample)
        # 'response', 'correct_answer'
    response_list = [sample['response'] for sample in data]
    correct_answers_list = [sample['correct_answer'] for sample in data]
    
    if task == 'GS_key':
        pass
    elif task == "MTT":
        # TODO: For each response, calculate t5/bert_score with all 50 tags an calculate the ROC-AUC and PR-AUC
        tags = list(np.load("data/MTT/tags.npy"))
        pass
    elif task == "EMO_valence":
        pass
    elif task == "EMO_arousal":
        pass
    elif task == "GTZAN":
        pass
    elif task == "VocalSet_tech":
        pass
    elif task == "Nsynth_instrument":
        pass
    elif task == "Nsynth_pitch":
        pass
    elif task == "ballroom_downbeat":
        pass
    elif task == 'gtzan_beat':
        pass
    elif task == "ballroom_beat":
        pass
    elif task == "gtzan_downbeat":
        pass
    elif task == "SDD":
        pass
    elif task == "MusicCaps":
        pass
    elif task == "DSing":
        pass
    elif task == "Guzheng_Tech":
        pass
    elif task == "MedleyDB":
        pass
    elif task == "MTG_instrument":
        # TODO: the same with MTT
        tags = ["accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass", "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", "drums", "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice"]
        pass
    elif task == "MTG_genre":
        # TODO: the same with MTT
        tags = ["60s", "70s", "80s", "90s", "acidjazz", "alternative", "alternativerock", "ambient", "atmospheric", "blues", "bluesrock", "bossanova", "breakbeat", "celtic", "chanson", "chillout", "choir", "classical", "classicrock", "club", "contemporary", "country", "dance", "darkambient", "darkwave", "deephouse", "disco", "downtempo", "drumnbass", "dub", "dubstep", "easylistening", "edm", "electronic", "electronica", "electropop", "ethno", "eurodance", "experimental", "folk", "funk", "fusion", "groove", "grunge", "hard", "hardrock", "hiphop", "house", "idm", "improvisation", "indie", "industrial", "instrumentalpop", "instrumentalrock", "jazz", "jazzfusion", "latin", "lounge", "medieval", "metal", "minimal", "newage", "newwave", "orchestral", "pop", "popfolk", "poprock", "postrock", "progressive", "psychedelic", "punkrock", "rap", "reggae", "rnb", "rock", "rocknroll", "singersongwriter", "soul", "soundtrack", "swing", "symphonic", "synthpop", "techno", "trance", "triphop", "world", "worldfusion"]
        pass
    elif task == "MTG_emotion":
        # TODO: the same with MTT
        tags = ["action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas", "commercial", "cool", "corporate", "dark", "deep", "documentary", "drama", "dramatic", "dream", "emotional", "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy", "happy", "heavy", "holiday", "hopeful", "inspiring", "love", "meditative", "melancholic", "melodic", "motivational", "movie", "nature", "party", "positive", "powerful", "relaxing", "retro", "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space", "sport", "summer", "trailer", "travel", "upbeat", "uplifting"]
        pass
    elif task == "MTG_top50tags":
        tags = ["alternative", "ambient", "atmospheric", "chillout", "classical", "dance", "downtempo", "easylistening", "electronic","experimental", "folk", "funk", "hiphop", "house", "indie", "instrumentalpop", "jazz", "lounge", "metal", "newage","orchestral", "pop", "popfolk", "poprock", "reggae", "rock", "soundtrack", 
                "techno","trance", "triphop","world", "acousticguitar", "bass", "computer", "drummachine", "drums", "electricguitar", "electricpiano", "guitar", "keyboard", "piano", "strings", "synthesizer", "violin", "voice", "emotional", "energetic", "film", "happy", "relaxing"]
        pass
    else:
        print("Task not found")

    
    
    
    
