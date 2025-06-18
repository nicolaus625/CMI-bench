import glob

import re
import argparse
import json
import os
import torch

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from bert_score import score

from torchmetrics import R2Score
from FlagEmbedding import FlagAutoModel
from torch.nn import functional as F

 
def normalise(text):
    if type(text) == list:
        text = text[0]
    return text.replace("_", "").replace("-", "").replace("#", "\u266f").replace("'", "").replace(" ", "").replace(".", "").lower()


def get_multiclass_acc(result_list):
    if type(result_list[0]["correct_answer"]) == list:
        answer_list  = set(tmp["correct_answer"][0] for tmp in result_list)
    else:
        answer_list = set(str(tmp["correct_answer"]) for tmp in result_list)
    # if type(data[0]["correct_answer"]) == str:
    length = len(set(answer_list))
    answer_list = [normalise(answer) for answer in answer_list]
    assert length == len(set(answer_list))
        
    # print(f"{len(answer_list)}-class classification")
    count = 0.0
    for tmp in result_list:
        reponse = normalise(tmp["response"])
        correct_answer = str(tmp["correct_answer"])
        if normalise(correct_answer) in reponse:
            # Ensure no other answer is in the response
            if all(answer not in reponse for answer in answer_list if answer != normalise(correct_answer)):
                count += 1
    return count / len(result_list)
    # elif type(data[0]["correct_answer"]) == int:
    #     # print(f"{len(answer_list)}-class classification")
    #     count = 0.0
    #     for tmp in result_list:
    #         if extract_int(tmp['response']) == tmp["correct_answer"]:
    #             count += 1
    #     return count / len(result_list)
    
def cal_r2(result_list):
    answer_list = [float(tmp["correct_answer"]) for tmp in result_list]

    # exception "score is 5 out of 9" -> 5
    response = [extract_int(re.sub(r'out of 9', '', tmp['response'])) 
                for tmp in result_list]

    response_2 = [x for x in response if x != -0.5]
    
    mean = np.mean(response_2)
    std = np.std(response_2)
    if std == 0:
        raise ValueError("Standard deviation is zero. Normalization not possible.")
    
    a = 1 / std
    b = -mean / std
    
    response_3 = [mean if x == -0.5 else x for x in response]
    normalised_response_3 = [(a * x + b) for x in response_3]

    r2score = R2Score()
    return r2score(torch.tensor(normalised_response_3), torch.tensor(answer_list))

def multi_label_classification(result_list, answer_list, subset_tags): # variable should not be called type otherwise it will override the built-in function
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

    if len(subset_tags) > 0:
        subset_tags = [ans.lower().strip() for ans in subset_tags]
        subset_indices = [i for i, tag in enumerate(answer_list) if tag in subset_tags]
        
        # STEP 4: compute ROC-AUC and PR-AUC for the subset
        for i in subset_indices:
            print(answer_list[i])
            y_true_subset = y_true[:, [i]]
            y_pred_subset = y_pred[:, [i]]
            subset_roc_auc = roc_auc_score(y_true_subset, y_pred_subset, average='macro')
            subset_pr_auc = average_precision_score(y_true_subset, y_pred_subset, average='macro')
            print(f"ROC: {100*subset_roc_auc:.2f}")
            print(f"PR: {100*subset_pr_auc:.2f}")

    # Calculate ROC-AUC and PR-AUC for each label
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    pr_auc = average_precision_score(y_true, y_pred, average='macro')

    return roc_auc, pr_auc


def multi_label_bert(result_list, answer_list, subset_tags, task="emotion", embed="bge"): 
    # Normalize predefined answer list
    answer_list = sorted([ans.lower().strip() for ans in answer_list])
    
    # Initialize lists to hold response vectors
    y_true = []
    y_pred = []
    
    model = FlagAutoModel.from_finetuned(
        "/map-vepfs/yinghao/huggingface/bge-large-en-v1.5",
        query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
        devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
    )
    
    from sentence_transformers import SentenceTransformer
    gte = SentenceTransformer("/map-vepfs/yinghao/huggingface/gte-Qwen2-7B-instruct", trust_remote_code=True)
    gte.max_seq_length = 8192
    
    for tmp in tqdm(result_list):
        response = tmp["response"].lower().strip()
        correct_answers = tmp["correct_answer"].lower().strip()
        
        # Create binary vector for true labels
        true_vector = [1 if answer in correct_answers else 0 for answer in answer_list]
        y_true.append(true_vector)
        
        if embed == "bert":
            bert_candidates = []
            bert_references = []
            # Store BERTScore inputs
            bert_candidates = [response] * len(answer_list)
            bert_references = answer_list
            # Compute BERTScore similarity
            P, R, F1 = score(bert_candidates, bert_references, lang="en", verbose=False)
            bert_scores = R.cpu().numpy()
            y_pred.append(bert_scores)
        elif embed == "bge":
            response_embed = torch.from_numpy(model.encode([response]))[0].view(1, -1)
            embeddings = torch.from_numpy(model.encode(answer_list))
            bge_cos = [
                F.cosine_similarity(response_embed,  embeddings[i].view(1, -1)).item()
                for i, andser in enumerate(answer_list)
            ]
        
            # Normalize BERT scores using softmax
            y_pred.append(bge_cos)
        elif embed == "gte":
            queries = [response]
            documents = answer_list
            query_embeddings = gte.encode(queries, prompt_name="query")
            document_embeddings = gte.encode(documents)

            scores = (query_embeddings @ document_embeddings.T) * 100
            y_pred.append(scores[0].tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if len(subset_tags) > 0:
        subset_tags = [ans.lower().strip() for ans in subset_tags]
        subset_indices = [i for i, tag in enumerate(answer_list) if tag in subset_tags]
        
        # STEP 4: compute ROC-AUC and PR-AUC for the subset
        for i in subset_indices:
            print(answer_list[i])
            y_true_subset = y_true[:, [i]]
            y_pred_subset = y_pred[:, [i]]
            subset_roc_auc = roc_auc_score(y_true_subset, y_pred_subset, average='macro')
            subset_pr_auc = average_precision_score(y_true_subset, y_pred_subset, average='macro')
            print(f"ROC: {100*subset_roc_auc:.2f}")
            print(f"PR: {100*subset_pr_auc:.2f}")
    
    # Compute standard ROC-AUC and PR-AUC
    roc_auc = roc_auc_score(y_true, y_pred, average='macro')
    pr_auc = average_precision_score(y_true, y_pred, average='macro')
    
    return {
        "ROC-AUC": roc_auc, 
        "PR-AUC": pr_auc, 
        # "BERT-Score (POC-AUC)": bert_scores.mean()
    }

CLASSES = ['Vibrato', 'Point Note', 'Upward Portamento', 'Downward Portamento', 'Plucks', 'Glissando', 'Tremolo']
CLASSES = [normalise(i) for i in CLASSES]
TOLERANCE = 0.05  # 50 ms onset tolerance

genre_set = {'singersongwriter', 'instrumentalrock', 'edm', 'newage', '70s', 'metal', 'alternative', 'punkrock', 'improvisation', 'worldfusion', 'country', 'progressive', 'rap', 'darkwave', 'house', 'alternativerock', 'rocknroll', 'lounge', 'grunge', 'bluesrock', 'orchestral', 'world', 'postrock', 'instrumentalpop', 'idm', 'folk', 'drumnbass', 'club', 'contemporary', 'chanson', 'deephouse', 'rnb', 'blues', 'popfolk', 'eurodance', 'electronica', 'electropop', 'latin', 'hardrock', 'celtic', 'easylistening', 'groove', 'trance', 'dubstep', 'soul', 'jazzfusion', 'atmospheric', 'downtempo', 'techno', 'hard', 'chillout', 'classicrock', 'darkambient', 'acidjazz', 'newwave', 'breakbeat', 'ethno', 'indie', '90s', 'electronic', 'dub', 'hiphop', 'bossanova', 'choir', 'minimal', 'soundtrack', 'triphop', 'synthpop', 'medieval', 'industrial', 'pop', 'swing', '80s', 'jazz', 'symphonic', 'psychedelic', 'dance', 'ambient', 'experimental', 'fusion', 'poprock', 'reggae', 'disco', '60s', 'rock', 'classical', 'funk'}
instrument_set = {'acousticguitar', 'saxophone', 'cello', 'strings', 'bass', 'bell', 'synthesizer', 'horn', 'keyboard', 'brass', 'harmonica', 'electricguitar', 'voice', 'bongo', 'guitar', 'harp', 'viola', 'pad', 'violin', 'drummachine', 'computer', 'orchestra', 'organ', 'drums', 'doublebass', 'percussion', 'acousticbassguitar', 'clarinet', 'trombone', 'accordion', 'rhodes', 'classicalguitar', 'trumpet', 'piano', 'oboe', 'flute', 'electricpiano', 'beat', 'sampler', 'pipeorgan'}
emotion_set = {'heavy', 'powerful', 'advertising', 'funny', 'motivational', 'sad', 'sexy', 'children', 'adventure', 'trailer', 'nature', 'christmas', 'energetic', 'fun', 'uplifting', 'inspiring', 'cool', 'party', 'relaxing', 'ballad', 'melancholic', 'drama', 'sport', 'film', 'romantic', 'commercial', 'love', 'dark', 'soundscape', 'background', 'summer', 'game', 'soft', 'epic', 'travel', 'slow', 'upbeat', 'positive', 'dramatic', 'space', 'deep', 'meditative', 'retro', 'documentary', 'calm', 'happy', 'emotional', 'dream', 'holiday', 'hopeful', 'groovy', 'melodic', 'fast', 'corporate', 'action', 'movie'}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="qwen2", type=str, 
                        choices=["qwen", "qwen2", "salmonn", "gpt-4o", "musilingo", "ltu", "ltu_as", "mullama", "flamingo", "gama", "gama_it", "pengi"], 
                        help='the model to use for inference')
    parser.add_argument('--task', default="MTT", type=str, 
                        choices=["all", "MTT", "EMO_valence", "EMO_arousal", "GTZAN", "VocalSet_tech", "Nsynth_instrument", "Nsynth_pitch", "ballroom_downbeat", "gtzan_beat", "ballroom_beat", "gtzan_downbeat", "SDD", "MusicCaps", "DSing", "Guzheng_Tech", "MedleyDB", "MTG_instrument", "MTG_genre", "GS_key", "MTG_emotion", "MTG_top50tags"], 
                        help='the task to evaluate')
    args = parser.parse_args()
    model = args.model 
    task = args.task
    results_json = glob.glob(f"model/results/{model}/{model}*.jsonl")
    if task != "all":
        results_json = [result for result in results_json if task in result]
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
    
    sub_genre = ["bossanova", "celtic", "chanson", "ethno", "latin", "medieval", "world", "worldfusion", "60s", "70s", "80s", "90s"]
    sub_instrument = ["bongo", "accordion", "harmonica", "piano", "violin"]
    
    if task == "MTT":
        tags = list(np.load("data/MTT/tags.npy"))
        # roc_auc, pr_auc = multi_label_classification(data, tags)
        # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
        gender = ['female', 'male', 'woman', 'male vocal','man', 'male voice', 'female vocal','female voice']
        value = multi_label_bert(data, tags, gender)
        print(f"{model}_{task} BGE\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # value = multi_label_bert(data, tags, embed="gte")
        # print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # value = multi_label_bert(data, tags, embed="bert")
        # print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    elif task == "MTG_instrument":
        tags = list(instrument_set)
        # roc_auc, pr_auc = multi_label_classification(data, tags)
        # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
        value = multi_label_bert(data, tags, sub_instrument)
        print(f"{model}_{task} BGE\n ROC-AUC: {value['ROC-AUC']*100:.2f}\n PR-AUC: {value['PR-AUC']*100:.2f}")
        # value = multi_label_bert(data, tags, embed="bert")
        # print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # value = multi_label_bert(data, tags, embed="gte")
        # print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # tags = ["accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass", "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", "drums", "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice"]
    elif task == "MTG_genre":
        tags = list(genre_set)
        # roc_auc, pr_auc = multi_label_classification(data, tags)
        # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
        value = multi_label_bert(data, tags, sub_genre )
        print(f"{model}_{task} BGE\n ROC-AUC: {100*value['ROC-AUC']:.2f}\n PR-AUC: {100*value['PR-AUC']:.2f}")
        # value = multi_label_bert(data, tags, embed="gte")
        # print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # value = multi_label_bert(data, tags, embed="bert")
        # print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        # tags = ["60s", "70s", "80s", "90s", "acidjazz", "alternative", "alternativerock", "ambient", "atmospheric", "blues", "bluesrock", "bossanova", "breakbeat", "celtic", "chanson", "chillout", "choir", "classical", "classicrock", "club", "contemporary", "country", "dance", "darkambient", "darkwave", "deephouse", "disco", "downtempo", "drumnbass", "dub", "dubstep", "easylistening", "edm", "electronic", "electronica", "electropop", "ethno", "eurodance", "experimental", "folk", "funk", "fusion", "groove", "grunge", "hard", "hardrock", "hiphop", "house", "idm", "improvisation", "indie", "industrial", "instrumentalpop", "instrumentalrock", "jazz", "jazzfusion", "latin", "lounge", "medieval", "metal", "minimal", "newage", "newwave", "orchestral", "pop", "popfolk", "poprock", "postrock", "progressive", "psychedelic", "punkrock", "rap", "reggae", "rnb", "rock", "rocknroll", "singersongwriter", "soul", "soundtrack", "swing", "symphonic", "synthpop", "techno", "trance", "triphop", "world", "worldfusion"]
    else:
        print(model, task)
        print("Task not found")

    
