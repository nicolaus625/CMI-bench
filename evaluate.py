import glob

import argparse
import json
import os
import torch

import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.tokenize import wordpunct_tokenize
from bert_score import score
import argparse

import mir_eval


 
def normalise(text):
    if type(text) == list:
        text = text[0]
    return text.replace("_", "").replace("-", "").replace("#", "\u266f").replace("'", "").replace(" ", "").replace(".", "").lower()

import re

def extract_int(response: str) -> int:
    """
    Extracts an integer from the given response string.
    Raises an error if more than one integer is found.

    Args:
        response (str): The input string containing a number.

    Returns:
        int: The extracted integer.

    Raises:
        ValueError: If more than one integer is found.
    """
    numbers = re.findall(r'\d+', response)  # Find all sequences of digits
    
    if not numbers:
        print("response:", response)
        # raise ValueError("No integer found in the response."
        print("No integer found in the response.")
        return -0.5
    elif len(numbers) > 1:
        print("response:", response)
        raise ValueError(f"Multiple integers found: {numbers}. Expected only one.")
    return int(numbers[0])  # Convert the first number to an integer


def get_multiclass_acc(result_list):
    if type(result_list[0]["correct_answer"]) == list:
        answer_list  = set(tmp["correct_answer"][0] for tmp in result_list)
    else:
        answer_list = set(tmp["correct_answer"] for tmp in result_list)
    if type(data[0]["correct_answer"]) == str:
        length = len(set(answer_list))
        answer_list = [normalise(answer) for answer in answer_list]
        assert length == len(set(answer_list))
        
        # print(f"{len(answer_list)}-class classification")
        count = 0.0
        for tmp in result_list:
            reponse = normalise(tmp["response"])
            if normalise(tmp["correct_answer"]) in reponse:
                # Ensure no other answer is in the response
                if all(answer not in reponse for answer in answer_list if answer != normalise(tmp["correct_answer"])):
                    count += 1
        return count / len(result_list)
    elif type(data[0]["correct_answer"]) == int:
        # print(f"{len(answer_list)}-class classification")
        count = 0.0
        for tmp in result_list:
            if extract_int(tmp['response']) == tmp["correct_answer"]:
                count += 1
        return count / len(result_list)

def multi_label_classification(result_list, answer_list): # variable should not be called type otherwise it will override the built-in function
    # Prepare answer_list as a flattened list of all possible answers
    answer_list = sorted([ans.lower().strip() for ans in answer_list])
        
    # Initialize a list to hold response vectors
    y_true = []
    y_pred = []

    for tmp in result_list:
        # Normalize responses and answers
        response = tmp["response"].lower().strip()
        # if task == "emotion":
        try:
            correct_answers = [normalise(ans) for ans in tmp["correct_answer"].split(",")]
        except:
            correct_answers = []
        
        # Create binary vectors for the true labels and predicted labels
        true_vector = [1 if answer in correct_answers else 0 for answer in answer_list]
        pred_vector = [1 if answer in response else 0 for answer in answer_list]
        
        y_true.append(true_vector)
        y_pred.append(pred_vector)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # print(y_true.shape, y_pred.shape)
    # print(y_true[:7])

    # Calculate ROC-AUC and PR-AUC for each label
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    pr_auc = average_precision_score(y_true, y_pred, average='macro')

    return roc_auc, pr_auc


def multi_label_bert(result_list, answer_list, task="emotion"): 
    # Normalize predefined answer list
    answer_list = sorted([ans.lower().strip() for ans in answer_list])
    
    # Initialize lists to hold response vectors
    y_true = []
    y_pred = []
    
    for tmp in result_list:
        response = tmp["response"].lower().strip()
        correct_answers = tmp["correct_answer"].lower().strip()
        
        # Create binary vector for true labels
        true_vector = [1 if answer in correct_answers else 0 for answer in answer_list]
        y_true.append(true_vector)
        
        bert_candidates = []
        bert_references = []
        
        # Store BERTScore inputs
        bert_candidates = [response] * len(answer_list)
        bert_references = answer_list
    
        # Compute BERTScore similarity
        P, R, F1 = score(bert_candidates, bert_references, lang="en", verbose=False)
        bert_scores = R.cpu().numpy()
    
        # Normalize BERT scores using softmax
        y_pred.append(bert_scores)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute standard ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    pr_auc = average_precision_score(y_true, y_pred, average='macro')
    
    return {
        "ROC-AUC": roc_auc, 
        "PR-AUC": pr_auc, 
        "BERT-Score (POC-AUC)": bert_scores.mean()
    }


def music_captioning(result_list):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score, bleu_score, bleu4_score, meteor_score = 0, 0, 0, 0
    mult_reference = []
    candidates = []
    for tmp in result_list:
        cand = tmp["response"]
        ref = tmp["correct_answer"]
        mult_reference.append([ref])
        candidates.append(cand)
        # print("ref",  ref)
        # print("cand", cand)
    
        rouge_score += scorer.score(ref, cand)['rougeL'].recall
        cand_split = wordpunct_tokenize(cand)
        ref_split = wordpunct_tokenize(ref)
        bleu4_score += sentence_bleu([ref], cand, weights=(0.0, 0.0, 0.0, 1.0))
        bleu_score += sentence_bleu([ref], cand)
        meteor_score += meteor_scorer([ref_split], cand_split)
        # break
    rouge_score, bleu_score, bleu4_score, meteor_score = rouge_score / (len(candidates)), bleu_score / (len(candidates)), bleu4_score / (len(candidates)), meteor_score / (len(candidates))
    P, R, F1 = score(candidates, mult_reference, lang="en", verbose=True)
    bert_score = R.mean().item()
    print(f"BLEU Score: {bleu_score}")
    print(f"BLEU-4 Score: {bleu4_score}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"BERT Score: {bert_score}")

def key_ensamble_score(result_list):
    def get_pred(tmp):
        classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        tmp = ''.join([i for i in tmp.lower().replace(" ","") 
                       if not i.isdigit()])
        if len(tmp) <=3 and tmp.endswith("m"):
            tmp = tmp + "inor"
        elif len(tmp) <=2:
            tmp = tmp + "major"
        map = {
            "c#": "db",
            "d#": "eb",
            "f#": "gb",
            "g#": "ab",
            "a#": "bb",
            "c♯": "db",
            "d♯": "eb",
            "f♯": "gb",
            "g♯": "ab",
            "a♯": "bb",
        }
        if tmp[1] in ["♯", "#"]:
            tmp = map[tmp[:2]] + tmp[2:]
        for class_ in classes:
            if class_.lower().strip().replace(" ","") in tmp:
                return class_
        return None
    score_list = []
    for tmp in result_list:
        try:
            score = mir_eval.key.weighted_score(
                    tmp["correct_answer"][0] if type(tmp["correct_answer"]) == list else tmp["correct_answer"], 
                    get_pred(tmp["response"])
                )
        except:
            if ',' not in tmp["response"]:
                print(tmp["correct_answer"], tmp["response"])
            score = None
        score_list.append(score if score is not None else 0)
    # print(score_list)
    return np.mean(score_list)

def beat_tracking(result_list, task="beat_tracking"):
    def get_beat(beats):
        if type(beats) == list:
            beats = beats[0]
        # print(beats, 
        #       "\n",
        #       beats.split("s")[:-1])
        tmp = []
        for i in beats.split(","):
            i = i.strip().replace("s", "")
            if len(i) > 0 and i[0].isdigit():
                if ":" not in i:
                    tmp.append(float(i.replace(",", "").strip()))
                elif i.split(":")[1] != "":
                    tmp.append(int(i.split(":")[0]) * 60 + float(i.split(":")[1]))
        tmp.sort()
        return np.array(
                tmp
        )
    results = []
    # f1_measure = np.mean([
    #     mir_eval.beat.f_measure(get_beat(tmp["correct_answer"]), 
    #                         get_beat(tmp["response"]))
    #     for tmp in result_list if tmp["task"] == "beat_tracking"
    # ])
    
    # CML_c, CML_t, AML_c, AML_t = mir_eval.beat.continuity(get_beat(result_list[0]["correct_answer"]), 
    #                         get_beat(result_list[0]["response"]))
    CML_c_values = []
    CML_t_values = []
    AML_c_values = []
    AML_t_values = []

    for tmp in result_list:
        try:
            results.append(
                mir_eval.beat.f_measure(get_beat(tmp["correct_answer"]), 
                        get_beat(tmp["response"])))
        except:
            # print(tmp["response"])
            results.append(0)

        try:
            CML_c, CML_t, AML_c, AML_t = mir_eval.beat.continuity(get_beat(tmp["correct_answer"]), get_beat(tmp["response"]))
        except:
            CML_c, CML_t, AML_c, AML_t = 0, 0, 0, 0
        CML_t_values.append(CML_t)
        AML_t_values.append(AML_t)

    f1_measure = np.mean(results)
    avg_CML_t = np.mean(CML_t_values)
    avg_AML_t = np.mean(AML_t_values)
    
    print(f"{task.upper()} F1: {f1_measure}")
    print(f"Average CMLt: {avg_CML_t}")
    print(f"Average AMLt: {avg_AML_t}")


genre_set = {'singersongwriter', 'instrumentalrock', 'edm', 'newage', '70s', 'metal', 'alternative', 'punkrock', 'improvisation', 'worldfusion', 'country', 'progressive', 'rap', 'darkwave', 'house', 'alternativerock', 'rocknroll', 'lounge', 'grunge', 'bluesrock', 'orchestral', 'world', 'postrock', 'instrumentalpop', 'idm', 'folk', 'drumnbass', 'club', 'contemporary', 'chanson', 'deephouse', 'rnb', 'blues', 'popfolk', 'eurodance', 'electronica', 'electropop', 'latin', 'hardrock', 'celtic', 'easylistening', 'groove', 'trance', 'dubstep', 'soul', 'jazzfusion', 'atmospheric', 'downtempo', 'techno', 'hard', 'chillout', 'classicrock', 'darkambient', 'acidjazz', 'newwave', 'breakbeat', 'ethno', 'indie', '90s', 'electronic', 'dub', 'hiphop', 'bossanova', 'choir', 'minimal', 'soundtrack', 'triphop', 'synthpop', 'medieval', 'industrial', 'pop', 'swing', '80s', 'jazz', 'symphonic', 'psychedelic', 'dance', 'ambient', 'experimental', 'fusion', 'poprock', 'reggae', 'disco', '60s', 'rock', 'classical', 'funk'}
instrument_set = {'acousticguitar', 'saxophone', 'cello', 'strings', 'bass', 'bell', 'synthesizer', 'horn', 'keyboard', 'brass', 'harmonica', 'electricguitar', 'voice', 'bongo', 'guitar', 'harp', 'viola', 'pad', 'violin', 'drummachine', 'computer', 'orchestra', 'organ', 'drums', 'doublebass', 'percussion', 'acousticbassguitar', 'clarinet', 'trombone', 'accordion', 'rhodes', 'classicalguitar', 'trumpet', 'piano', 'oboe', 'flute', 'electricpiano', 'beat', 'sampler', 'pipeorgan'}
emotion_set = {'heavy', 'powerful', 'advertising', 'funny', 'motivational', 'sad', 'sexy', 'children', 'adventure', 'trailer', 'nature', 'christmas', 'energetic', 'fun', 'uplifting', 'inspiring', 'cool', 'party', 'relaxing', 'ballad', 'melancholic', 'drama', 'sport', 'film', 'romantic', 'commercial', 'love', 'dark', 'soundscape', 'background', 'summer', 'game', 'soft', 'epic', 'travel', 'slow', 'upbeat', 'positive', 'dramatic', 'space', 'deep', 'meditative', 'retro', 'documentary', 'calm', 'happy', 'emotional', 'dream', 'holiday', 'hopeful', 'groovy', 'melodic', 'fast', 'corporate', 'action', 'movie'}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="qwen2", type=str, 
                        choices=["qwen", "qwen2", "salmonn", "gpt-4o", "musilingo", "ltu", "ltuas", "mullama", "flamingo", "gama", "gama_it"], 
                        help='the model to use for inference')
    
    args = parser.parse_args()
    model = args.model 
    results_json = glob.glob(f"model/results_test/{model}/{model}*.jsonl")
    results_json = [result for result in results_json if "MTG_instrument" in result]
    result = results_json[0]
    task = os.path.basename(result)[len(model)+1:-6]
    # load jsonl
    with open(result, "r") as f:
        # data = [json.load(line.strip()) for line in f]
        data = json.load(f)
    f.close()
    
    for sample in data:
        print("sample", sample)
        break
        # 'response', 'correct_answer'
    response_list = [sample['response'] for sample in data]
    correct_answers_list = [sample['correct_answer'] for sample in data]
    
    if task == 'GS_key':
        gmean_score = key_ensamble_score(data)
        print(f"{model}_{task} G-Mean: {gmean_score:.4f}")
    elif task == "MTT":
        tags = list(np.load("data/MTT/tags.npy"))
        value = multi_label_classification(data, tags)
        print(f"{model}_{task} Accurate\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        value = multi_label_bert(data, tags)
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    elif task == "EMO_valence":
        print(model, task)
        pass
    elif task == "EMO_arousal":
        print(model, task)
        pass
    elif task == "GTZAN":
        acc = get_multiclass_acc(data)
        print(f"{model}_{task} genre Acc: {acc:.4f}")
    elif task == "VocalSet_tech":
        acc = get_multiclass_acc(data)
        print(f"{model}_{task} Acc: {acc:.4f}")
    elif task == "Nsynth_instrument":
        instrument_list = [tmp for tmp in data]
        acc = get_multiclass_acc(instrument_list)
        print(f"{model}_{task} Acc: {acc:.4f}")
    elif task == "Nsynth_pitch":
        acc = get_multiclass_acc(data)
        print(f"{model}_{task} Acc: {acc:.4f}")
    elif task == "ballroom_downbeat":
        beat_tracking(data, task="downbeat_tracking")
    elif task == 'gtzan_beat':
        beat_tracking(data)
    elif task == "ballroom_beat":
        beat_tracking(data)
    elif task == "gtzan_downbeat":
        beat_tracking(data, task="downbeat_tracking")
    elif task == "SDD":
        music_captioning(data)
    elif task == "MusicCaps":
        music_captioning(data)
    elif task == "DSing":
        print(model, task)
        pass
    elif task == "Guzheng_Tech":
        print(model, task)
        pass
    elif task == "MedleyDB":
        print(model, task)
        pass
    elif task == "MTG_instrument":
        tags = list(instrument_set)
        value = multi_label_classification(data, tags)
        print(f"{model}_{task} Accurate\n ROC-AUC: {value[0]:.4f}\n PR-AUC: {value[1]:.4f}")
        value = multi_label_bert(data, tags)
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # tags = ["accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass", "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", "drums", "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice"]
    elif task == "MTG_genre":
        tags = list(genre_set)
        value = multi_label_classification(data, tags)
        print(f"{model}_{task} Accurate\n ROC-AUC: {value[0]:.4f}\n PR-AUC: {value[1]:.4f}")
        value = multi_label_bert(data, tags)
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # tags = ["60s", "70s", "80s", "90s", "acidjazz", "alternative", "alternativerock", "ambient", "atmospheric", "blues", "bluesrock", "bossanova", "breakbeat", "celtic", "chanson", "chillout", "choir", "classical", "classicrock", "club", "contemporary", "country", "dance", "darkambient", "darkwave", "deephouse", "disco", "downtempo", "drumnbass", "dub", "dubstep", "easylistening", "edm", "electronic", "electronica", "electropop", "ethno", "eurodance", "experimental", "folk", "funk", "fusion", "groove", "grunge", "hard", "hardrock", "hiphop", "house", "idm", "improvisation", "indie", "industrial", "instrumentalpop", "instrumentalrock", "jazz", "jazzfusion", "latin", "lounge", "medieval", "metal", "minimal", "newage", "newwave", "orchestral", "pop", "popfolk", "poprock", "postrock", "progressive", "psychedelic", "punkrock", "rap", "reggae", "rnb", "rock", "rocknroll", "singersongwriter", "soul", "soundtrack", "swing", "symphonic", "synthpop", "techno", "trance", "triphop", "world", "worldfusion"]
    elif task == "MTG_emotion":
        tags = list(emotion_set)
        value = multi_label_classification(data, tags)
        print(f"{model}_{task} Accurate\n ROC-AUC: {value[0]:.4f}\n PR-AUC: {value[1]:.4f}")
        value = multi_label_bert(data, tags)
        print(f"{model}_{task}\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # tags = ["action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas", "commercial", "cool", "corporate", "dark", "deep", "documentary", "drama", "dramatic", "dream", "emotional", "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy", "happy", "heavy", "holiday", "hopeful", "inspiring", "love", "meditative", "melancholic", "melodic", "motivational", "movie", "nature", "party", "positive", "powerful", "relaxing", "retro", "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space", "sport", "summer", "trailer", "travel", "upbeat", "uplifting"]
    elif task == "MTG_top50tags":
        tags = ["alternative", "ambient", "atmospheric", "chillout", "classical", "dance", "downtempo", "easylistening", "electronic","experimental", "folk", "funk", "hiphop", "house", "indie", "instrumentalpop", "jazz", "lounge", "metal", "newage","orchestral", "pop", "popfolk", "poprock", "reggae", "rock", "soundtrack", 
                "techno","trance", "triphop","world", "acousticguitar", "bass", "computer", "drummachine", "drums", "electricguitar", "electricpiano", "guitar", "keyboard", "piano", "strings", "synthesizer", "violin", "voice", "emotional", "energetic", "film", "happy", "relaxing"]
        value = multi_label_classification(data, tags)
        print(f"{model}_{task} Accurate\n ROC-AUC: {value[0]:.4f}\n PR-AUC: {value[1]:.4f}")
        value = multi_label_bert(data, tags)
        print(f"{model}_{task}\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    else:
        print(model, task)
        print("Task not found")

    
    
    
    
