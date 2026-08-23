import glob

import ast
import unicodedata
import re
import argparse
import json
import os
import string
import torch

MODEL_CACHE_DIR = os.environ.get(
    "CMI_BENCH_MODEL_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "cmi-bench", "models"),
)
HF_CACHE_DIR = os.path.join(MODEL_CACHE_DIR, ".cache", "huggingface")
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "transformers"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(HF_CACHE_DIR, "sentence_transformers"))

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.tokenize import wordpunct_tokenize
from bert_score import score

import mir_eval
from torchmetrics import R2Score
from num2words import num2words
import jiwer
import pretty_midi
from torch.nn import functional as F

 
def normalise(text):
    if type(text) == list:
        text = text[0]
    return text.replace("_", "").replace("-", "").replace("#", "\u266f").replace("'", "").replace(" ", "").replace(".", "").lower()

def clean_response(response):
    if not isinstance(response, str):
        return response
    markers = [
        "\nassistant\n",
        "\nassistant:",
        "<|im_start|>assistant",
    ]
    for marker in markers:
        if marker in response:
            response = response.split(marker)[-1]
    response = response.replace("<|im_end|>", "")
    return response.strip()

def resolve_hf_model(local_dir_name, repo_id):
    local_path = os.path.join(MODEL_CACHE_DIR, local_dir_name)
    if os.path.isdir(local_path) and os.listdir(local_path):
        return local_path
    print(
        f"Local model not found at {local_path}. "
        f"Falling back to Hugging Face repo {repo_id}; downloads will use {HF_CACHE_DIR}."
    )
    return repo_id


def apply_gte_compat_patches():
    # The installed flash-attn extension was built against an older torch ABI.
    # GTE-Qwen2 can run without it, so keep evaluation on the SDPA/eager path.
    try:
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
    except Exception:
        return

    import_utils.is_flash_attn_2_available = lambda: False
    import_utils.is_flash_attn_3_available = lambda: False
    transformers_utils.is_flash_attn_2_available = lambda: False
    transformers_utils.is_flash_attn_3_available = lambda: False

    # The cached GTE-Qwen2 remote code was written for an older transformers
    # Cache API. Newer transformers removed get_usable_length in favor of
    # get_seq_length/get_max_cache_shape.
    try:
        from transformers.cache_utils import Cache, DynamicCache
    except Exception:
        return

    if not hasattr(Cache, "get_usable_length"):
        def get_usable_length(self, new_seq_length=None, layer_idx=0):
            previous_seq_length = self.get_seq_length(layer_idx)
            max_cache_shape = self.get_max_cache_shape()
            if (
                max_cache_shape is not None
                and max_cache_shape > 0
                and new_seq_length is not None
                and previous_seq_length + new_seq_length > max_cache_shape
            ):
                return max_cache_shape - new_seq_length
            return previous_seq_length

        Cache.get_usable_length = get_usable_length

    if not hasattr(DynamicCache, "get_usable_length"):
        DynamicCache.get_usable_length = Cache.get_usable_length


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
    response = response.replace("1-9 scale", "scale")  # remove scale number
    response = response.replace("from 1 to 9", "scale")  # remove scale number
    response = re.sub(r'\d-point scale', 'scale', response) # delete r'\d-scale' in the response
    response = re.sub(r'\d/\d time signature', 'time signature', response) # delete r'\d-scale' in the response
    response = re.sub(r'\d/\d beat', 'time signature', response) # delete r'\d-scale' in the response
    response = re.sub(r'Example \d', '', response) # delete r'\d-scale' in the response
    numbers = re.findall(r'\d+', response)  # Find all sequences of digits

    numbers = [i for i in numbers if 0 < int(i) < 10]  # Filter out numbers outside the range 1-9, such as bpm
    if not numbers:
        # print("No integer found in the response:", response)
        # raise ValueError("No integer found in the response."
        return -0.5
    elif len(numbers) > 1:
        if len(numbers) == 2 and f"{numbers[0]}.{numbers[1]}" in response:  # typical for flamingo
            return float(f"{numbers[0]}.{numbers[1]}")
        # eg1, around 8 and 9 -> 8, eg2. 7, becase xxx, so 7 ->7.  eg3. score is 5.2 -> 5
        if len(numbers) == 2 and numbers[0] == numbers[1]:
            return int(numbers[0])
        return int(numbers[-1])
        # raise ValueError(f"Multiple integers found: {numbers}. Expected only one.")
    return int(numbers[0])  # Convert the first number to an integer


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
    if len(response_2) == 0:
        response_2 = [0.0]
    
    mean = np.mean(response_2)
    std = np.std(response_2)
    if std == 0:
        normalised_response_3 = [0.0 for _ in response]
    else:
        a = 1 / std
        b = -mean / std

        response_3 = [mean if x == -0.5 else x for x in response]
        normalised_response_3 = [(a * x + b) for x in response_3]

    r2score = R2Score()
    return r2score(torch.tensor(normalised_response_3), torch.tensor(answer_list))

def safe_macro_roc_auc(y_true, y_pred):
    """Compute macro ROC-AUC over labels that have both classes present."""
    scores = []
    for label_idx in range(y_true.shape[1]):
        label_true = y_true[:, label_idx]
        if np.unique(label_true).size < 2:
            continue
        scores.append(roc_auc_score(label_true, y_pred[:, label_idx]))
    return float(np.mean(scores)) if scores else float("nan")

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
    roc_auc = safe_macro_roc_auc(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred, average='macro')

    return roc_auc, pr_auc


def lyra_label_set(value):
    if isinstance(value, list):
        values = value
    else:
        values = str(value).split(",")
    return {normalise(item) for item in values if normalise(item)}


def lyra_multilabel_scores(result_list):
    answer_list = sorted(
        {
            label
            for sample in result_list
            for label in lyra_label_set(sample["correct_answer"])
        }
    )
    true_vectors = []
    pred_vectors = []
    exact_matches = []

    for sample in result_list:
        response = normalise(sample["response"])
        correct = lyra_label_set(sample["correct_answer"])
        predicted = {label for label in answer_list if label in response}

        true_vectors.append([1 if label in correct else 0 for label in answer_list])
        pred_vectors.append([1 if label in predicted else 0 for label in answer_list])
        exact_matches.append(predicted == correct)

    y_true = np.array(true_vectors)
    y_pred = np.array(pred_vectors)

    tp = np.logical_and(y_true == 1, y_pred == 1).sum()
    fp = np.logical_and(y_true == 0, y_pred == 1).sum()
    fn = np.logical_and(y_true == 1, y_pred == 0).sum()
    micro_f1 = 0.0 if 2 * tp + fp + fn == 0 else (2 * tp) / (2 * tp + fp + fn)

    per_label_f1 = []
    for idx in range(len(answer_list)):
        label_tp = np.logical_and(y_true[:, idx] == 1, y_pred[:, idx] == 1).sum()
        label_fp = np.logical_and(y_true[:, idx] == 0, y_pred[:, idx] == 1).sum()
        label_fn = np.logical_and(y_true[:, idx] == 1, y_pred[:, idx] == 0).sum()
        denom = 2 * label_tp + label_fp + label_fn
        per_label_f1.append(0.0 if denom == 0 else (2 * label_tp) / denom)

    return {
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(per_label_f1)) if per_label_f1 else 0.0,
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "num_labels": len(answer_list),
    }


def lyra_dance_acc(result_list):
    correct = 0
    for sample in result_list:
        response = normalise(sample["response"])
        answer = normalise(sample["correct_answer"])
        if answer == "yes":
            predicted = "yes" in response and "no" not in response
        else:
            predicted = "no" in response and "yes" not in response
        correct += int(predicted)
    return correct / len(result_list)


def andalusian_normalise(text):
    text = normalise(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    replacements = {
        "ā": "a",
        "ī": "i",
        "ū": "u",
        "ḥ": "h",
        "ḍ": "d",
        "ṣ": "s",
        "ṭ": "t",
        "‘": "",
        "’": "",
        "`": "",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def andalusian_multiclass_acc(result_list):
    count = 0
    for sample in result_list:
        response = andalusian_normalise(sample["response"])
        correct = andalusian_normalise(sample["correct_answer"])
        if correct and correct in response:
            count += 1
    return count / len(result_list)


def multi_label_bert(result_list, answer_list, task="emotion", embed="bge"): 
    # Normalize predefined answer list
    answer_list = sorted([ans.lower().strip() for ans in answer_list])
    
    # Initialize lists to hold response vectors
    y_true = []
    y_pred = []
    
    model = None
    gte = None
    if embed == "bge":
        from FlagEmbedding import FlagAutoModel

        bge_model = resolve_hf_model("bge-large-en-v1.5", "BAAI/bge-large-en-v1.5")
        model = FlagAutoModel.from_finetuned(
            bge_model,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
        )
        embeddings = torch.from_numpy(model.encode(answer_list))
    elif embed == "gte":
        apply_gte_compat_patches()
        from sentence_transformers import SentenceTransformer

        gte_model = resolve_hf_model("gte-Qwen2-7B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct")
        gte = SentenceTransformer(
            gte_model,
            trust_remote_code=True,
            cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"],
        )
        gte.max_seq_length = 8192
        document_embeddings = gte.encode(answer_list)
    
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
            bge_cos = [
                F.cosine_similarity(response_embed,  embeddings[i].view(1, -1)).item()
                for i, andser in enumerate(answer_list)
            ]
        
            # Normalize BERT scores using softmax
            y_pred.append(bge_cos)
        elif embed == "gte":
            queries = [response]
            query_embeddings = gte.encode(queries, prompt_name="query")

            scores = (query_embeddings @ document_embeddings.T) * 100
            y_pred.append(scores[0].tolist())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute standard ROC-AUC and PR-AUC
    roc_auc = safe_macro_roc_auc(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred, average='macro')
    
    return {
        "ROC-AUC": roc_auc, 
        "PR-AUC": pr_auc, 
        # "BERT-Score (POC-AUC)": bert_scores.mean()
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
    
    print(f"{task.upper()} F1: {f1_measure*100:.2f}")
    print(f"Average CMLt: {avg_CML_t*100:.2f}")
    print(f"Average AMLt: {avg_AML_t*100:.2f}")

def convert_digits_to_words(words):
    for i, word in enumerate(words):
        if word.isdigit():
            words[i] = num2words(int(word))
    return words

def compute_wer_cer(prediction, reference):
    # Clean the prediction (remove prefix)
    patterns = [
        r".*? lyrics .*?are.*?:",
        r".*? content .*?is.*?:",
        r".*? transcription .*?is.*?:",
        r".*? text .*?is.*?:",
        "<s>",
        "</s>"
    ]
    for pattern in patterns:
        prediction = re.sub(pattern, '', prediction).strip()

    def clean_string(text):
        # text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.translate(str.maketrans('', '', '!"#$%&\()*+,-./:;<=>?@[\\]^_`{|}~'))  # remain \'
        text = text.lower().replace("\n", " ")
        text = convert_digits_to_words(text.split())
        text = " ".join(text)
        return text
    prediction = clean_string(prediction)
    reference = clean_string(reference)
    
    # Compute WER and CER using jiwer
    wer = jiwer.wer(reference, prediction)
    cer = jiwer.cer(reference, prediction)
    
    return wer, cer

def batch_wer_cer(result_list):
    predictions = [tmp["response"] for tmp in result_list]
    references = [tmp["correct_answer"] for tmp in result_list]
    wer_scores = []
    cer_scores = []
    
    for prediction, reference in zip(predictions, references):
        wer, cer = compute_wer_cer(prediction, reference)
        wer_scores.append(wer)
        cer_scores.append(cer)
    
    return np.mean(wer_scores), np.mean(cer_scores)


def process_midi_sequence(input_string):
    # Step 1: Check if input is a valid string and parse it
    if not isinstance(input_string, str):
        return None, None
        # raise ValueError("Input is not a string")
    if "{" in input_string or "}" in input_string:
        input_string = re.sub(r"{'time':", "(", input_string)
        input_string = re.sub(r"'MIDI_number':", "", input_string)
        input_string = re.sub(r"}", ")", input_string)
    if "{" in input_string or "}" in input_string:
        return None, None
        # raise ValueError("Invalid characters in input string")
    
    input_string = input_string.replace("♯","#")
    input_string = input_string.replace("♭","b")
    if len(input_string) < 2:
        return None, None
        # raise ValueError("Empty input string")
    input_string = input_string[:-1] if input_string[-2]=="]" else input_string  #punctuation after "[]"
    try:
        midi_sequence = eval(input_string)
    except (SyntaxError, NameError, TypeError) as e:
        if 'unterminated string literal' in str(e):
            last_paren = input_string.rfind(')')
            start_paren = input_string.find('[', 0, last_paren)
            if start_paren == -1:
                return None, None
            fixed_string = input_string[start_paren:last_paren] + ")]"
            try:
                midi_sequence = eval(fixed_string)
            except Exception as inner_e:
                return None, None
        elif "'[' was never closed" in str(e):
            try:
                midi_sequence = eval( input_string + "]")
            except Exception as inner_e:
                return None, None
        else:
            # input_str = '[(0.52, ), (0.67, 德音), (0.83, 음7), (1.00, C♯6), (1.14, D7), (1.30, F7)]'
            matches = re.findall(r'\(([^)]+)\)', input_string)
            try:
                midi_sequence = [(match.split(",")[0], match.split(",")[1]) for match in matches]
            except:
                # print(input_string)
                return None, None
    
    if not isinstance(midi_sequence, list) or not all(isinstance(item, tuple) and len(item) == 2 for item in midi_sequence):
        return None, None
        # raise ValueError("Invalid format after eval")
    
    for idx, item in enumerate(midi_sequence):
        if isinstance(item[0], float):
            continue
        # exception:2:43, '0.00'
        if item[0].startswith("\'") and item[0].endswith("\'"):
            midi_sequence[idx] = (item[1:-1], item[1])
        if ":" in str(item[0]):
            try:
                midi_sequence[idx] = (float(item[0].split(":")[1]) + float(item[0].split(":")[0]) * 60, item[1])
            except:
                # print(midi_sequence)  "time:"
                return None, None
    try:
        midi_sequence = [(float(x[0]), x[1]) for x in midi_sequence]
    except:
        # print(midi_sequence)  [('7766 / 1000', ' 10')]
        return None, None
    midi_sequence = sorted(midi_sequence, key=lambda x: x[0])
    seen = {}
    for item in midi_sequence:
        if item[0] not in seen:
            seen[item[0]] = item
    try:
        midi_sequence = sorted(seen.values(), key=lambda x: float(x[0])) # str/formula to float
    except:
        # print(midi_sequence)
        return None, None
    if len(midi_sequence) == 0:
        return None, None
    
    midi_array = np.array(midi_sequence, dtype=object)
    shift = float(midi_array[0, 0]) // 10 * 10
    midi_array[:, 0] = np.array([float(x) - shift for x in midi_array[:, 0]])

    # Step 4: Convert note names to MIDI numbers
    for i, note in enumerate(midi_array[:, 1]):
        if note != '':  # If it's a note name, convert it
            if isinstance(note, str):  # If it's a note name, convert it
                try:
                    midi_array[i, 1] = float(midi_array[i, 1])
                except:
                    try:
                        midi_array[i, 1] = pretty_midi.note_name_to_number(note)
                        # raise ValueError(f"Invalid MIDI note name '{note}': {e}")
                    except:  # note might be a string with Chinese characters which is not a midi name
                        midi_array[i, 1] = 0
            midi_array[i, 1] = pretty_midi.note_number_to_hz(midi_array[i, 1])
        else:  # 0Hz, not midi_num=0
            midi_array[i, 1] = 0.0

    # Convert dtype to float after processing
    midi_array = midi_array.astype(float)
    time = midi_array[:, 0]
    frequency = midi_array[:, 1]

    return time, frequency


def melody_evaluation(result_list):
    # Initialize lists to hold response vectors
    overall_accuracy = []

    for tmp in result_list:
        # Normalize responses and answers
        response = tmp["response"]
        correct_answers = tmp["correct_answer"]

        # Process MIDI sequences
        try:
            response_time, response_freq = process_midi_sequence(response)
            correct_time, correct_freq = process_midi_sequence(correct_answers)
        except Exception:
            overall_accuracy.append(0)
            continue
        
        if response_time is None or correct_time is None:
            overall_accuracy.append(0)
            continue
        
        try:
            overall_accuracy.append(
                mir_eval.melody.evaluate(correct_time, correct_freq, 
                                        response_time, response_freq)['Overall Accuracy']
            )
        except:
            print(tmp)

    return np.mean(overall_accuracy)


from sklearn.metrics import f1_score

CLASSES = ['Vibrato', 'Point Note', 'Upward Portamento', 'Downward Portamento', 'Plucks', 'Glissando', 'Tremolo']
CLASSES = [normalise(i) for i in CLASSES]
TOLERANCE = 0.05  # 50 ms onset tolerance

def convert_to_frame_labels(events, sr=100):
    """
    Convert a list of (start_time, end_time, class) events into frame-based labels.
    Args:
        events: list of (start_time, end_time, class) tuples
        sr: frame rate in Hz (default = 100 for 10 ms frame step)
    Returns:
        frame_labels: np.ndarray of shape (num_frames, num_classes)
    """
    start, end = events.find('['), events.find(')')
    if start == -1 or start >= end:
        events = [('0','10','No Tech')]
    else:
        # should only have one "]" if following instruction
        try:
            events_string = events[start:end ] + ")]"
            if events_string.startswith("[\"["):
                events_string = events_string[2:]
            events = eval(events_string)
        except:
            events = [('0','10','No Tech')]
    if len(events) == 0:
        events = [('0','10','No Tech')]
    if isinstance(events, list) and isinstance(events[0], dict):
        events = [(float(e['start']), float(e['end']), e['technique']) for e in events]
    elif isinstance(events, list) and all(isinstance(e, tuple) and len(e) == 3 for e in events):
        try:
            events = [(float(e[0]), float(e[1]), str(e[2])) for e in events if e[0][0].isdigit()]
            events = [(0,10,'No Tech')] if len(events)==0 else events
        except:
            events = [(0,10,'No Tech')]
    try:
        events = [(start %10, (end -1e-4) %10 + 1e-4, normalise(label)) for start, end, label in events]
    except:
        events = [(0,10,'notech')]

    max_time = 10 #max(float(event[1]) for event in events) if events else 0
    num_frames = int(np.ceil(max_time * sr))
    frame_labels = np.zeros((num_frames, len(CLASSES)))
    if len(events) == 1 and normalise(events[0][-1]) == "notech":
        return frame_labels
    
    for event in events:
        start_frame = int(float(event[0]) * sr)
        end_frame = int(float(event[1]) * sr)
        label = normalise(event[2])
        if label not in CLASSES:
            continue
        label_idx = CLASSES.index(label)
        frame_labels[start_frame:end_frame, label_idx] = 1
        
    return frame_labels


def calculate_frame_f1(result_list, sr=100):
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    tp_per_class = np.zeros(len(CLASSES))
    fp_per_class = np.zeros(len(CLASSES))
    fn_per_class = np.zeros(len(CLASSES))
    
    for tmp in result_list:
        true_events = tmp["correct_answer"]
        pred_events = tmp["response"]

        y_true = convert_to_frame_labels(true_events, sr)
        y_pred = convert_to_frame_labels(pred_events, sr)

        # Accumulate micro F1 components
        total_tp += ((y_true == 1) & (y_pred == 1)).sum()
        total_fp += ((y_true == 0) & (y_pred == 1)).sum()
        total_fn += ((y_true == 1) & (y_pred == 0)).sum()
        total_tn += ((y_true == 0) & (y_pred == 0)).sum()

        # Accumulate per-class components for macro-F1
        for i in range(len(CLASSES)):
            tp_per_class[i] += ((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum()
            fp_per_class[i] += ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
            fn_per_class[i] += ((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum()

    # Micro-F1 calculation
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    # Macro-F1 calculation
    class_f1 = []
    for i in range(len(CLASSES)):
        precision = tp_per_class[i] / (tp_per_class[i] + fp_per_class[i]) if (tp_per_class[i] + fp_per_class[i]) > 0 else 0
        recall = tp_per_class[i] / (tp_per_class[i] + fn_per_class[i]) if (tp_per_class[i] + fn_per_class[i]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_f1.append(f1)
    
    macro_f1 = np.mean(class_f1)

    return micro_f1, macro_f1


def parse_song_structure_time(value):
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip().lower()
    text = (
        text.replace("seconds", "")
        .replace("second", "")
        .replace("secs", "")
        .replace("sec", "")
        .strip()
    )
    if text.endswith("s"):
        text = text[:-1].strip()

    if ":" in text:
        parts = text.split(":")
        try:
            return sum(float(part) * (60 ** idx) for idx, part in enumerate(reversed(parts)))
        except ValueError:
            return None

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    return float(match.group(0))


def normalise_song_structure_label(label):
    label = str(label).strip().lower()
    label = re.sub(r"\s+", " ", label)
    label = label.replace("_", "-")
    label = label.strip(" .,:;")

    label_map = {
        "pre chorus": "pre-chorus",
        "prechorus": "pre-chorus",
        "post chorus": "chorus",
        "postchorus": "chorus",
        "post-chorus": "chorus",
        "refrain": "chorus",
        "instrumental": "inst",
        "interlude": "inst",
        "instrumental break": "inst",
        "silent": "silence",
    }
    if label in label_map:
        return label_map[label]

    known_labels = [
        "pre-chorus",
        "chorus",
        "verse",
        "bridge",
        "intro",
        "outro",
        "inst",
        "silence",
    ]
    for known_label in known_labels:
        if re.search(rf"(^|[^a-z]){re.escape(known_label)}([^a-z]|$)", label):
            return known_label

    return "unknown" if label else "unknown"


def coerce_song_structure_segments(raw_segments):
    segments = []
    if isinstance(raw_segments, dict):
        raw_segments = raw_segments.get("segments", raw_segments.get("labels", []))

    if not isinstance(raw_segments, list):
        return segments

    for item in raw_segments:
        if isinstance(item, str):
            nested = parse_song_structure_segments(item)
            segments.extend(nested)
            continue

        if isinstance(item, dict):
            start = item.get("start", item.get("start_time", item.get("onset")))
            end = item.get("end", item.get("end_time", item.get("offset")))
            label = item.get("label", item.get("section", item.get("type")))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
            label = item[2] if len(item) >= 3 else "unknown"
        else:
            continue

        start = parse_song_structure_time(start)
        end = parse_song_structure_time(end)
        if start is None or end is None or end <= start or label is None:
            continue
        segments.append((start, end, normalise_song_structure_label(label)))

    return sorted(segments, key=lambda item: (item[0], item[1]))


def parse_song_structure_segments(text):
    if isinstance(text, list):
        return coerce_song_structure_segments(text)
    if not isinstance(text, str):
        return []

    text = text.strip()
    candidates = [text]
    list_start = text.find("[")
    list_end = text.rfind("]")
    if list_start != -1 and list_end > list_start:
        candidates.append(text[list_start:list_end + 1])

    for candidate in candidates:
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        segments = coerce_song_structure_segments(parsed)
        if segments:
            return segments

    pattern = re.compile(
        r"\(?\s*['\"]?([0-9][0-9:.]*\s*s?)['\"]?\s*,\s*"
        r"['\"]?([0-9][0-9:.]*\s*s?)['\"]?\s*,\s*"
        r"['\"]?([^'\"\)\]\n,]+)['\"]?\s*\)?"
    )
    segments = []
    for match in pattern.finditer(text):
        start = parse_song_structure_time(match.group(1))
        end = parse_song_structure_time(match.group(2))
        label = normalise_song_structure_label(match.group(3))
        if start is not None and end is not None and end > start:
            segments.append((start, end, label))

    if segments:
        return sorted(segments, key=lambda item: (item[0], item[1]))

    pair_pattern = re.compile(
        r"\(?\s*['\"]?([0-9][0-9:.]*\s*s?)['\"]?\s*,\s*"
        r"['\"]?([0-9][0-9:.]*\s*s?)['\"]?\s*\)?"
    )
    for match in pair_pattern.finditer(text):
        start = parse_song_structure_time(match.group(1))
        end = parse_song_structure_time(match.group(2))
        if start is not None and end is not None and end > start:
            segments.append((start, end, "unknown"))

    if segments:
        return sorted(segments, key=lambda item: (item[0], item[1]))

    bracketed_range_pattern = re.compile(
        r"[\[\(]\s*([0-9][0-9:.]*\s*s?)\s*-\s*"
        r"([0-9][0-9:.]*\s*s?)\s*[\]\)]\s*"
        r"([^,\n\]\)]+)?",
        re.IGNORECASE,
    )
    for match in bracketed_range_pattern.finditer(text):
        start = parse_song_structure_time(match.group(1))
        end = parse_song_structure_time(match.group(2))
        label = normalise_song_structure_label(match.group(3) or "unknown")
        if start is not None and end is not None and end > start:
            segments.append((start, end, label))

    if segments:
        return sorted(segments, key=lambda item: (item[0], item[1]))

    bare_range_pattern = re.compile(
        r"['\"]?\s*([0-9][0-9:.]*\s*s?)\s*-\s*"
        r"([0-9][0-9:.]*\s*s?)\s*['\"]?",
        re.IGNORECASE,
    )
    for match in bare_range_pattern.finditer(text):
        start = parse_song_structure_time(match.group(1))
        end = parse_song_structure_time(match.group(2))
        if start is not None and end is not None and end > start:
            segments.append((start, end, "unknown"))

    return sorted(segments, key=lambda item: (item[0], item[1]))


def segments_to_intervals_and_labels(segments, start_time=None, end_time=None):
    intervals = []
    labels = []
    cursor = start_time if start_time is not None else segments[0][0]

    for start, end, label in segments:
        if start_time is not None:
            start = max(start, start_time)
        if end_time is not None:
            end = min(end, end_time)
        if end <= cursor:
            continue
        if start > cursor:
            intervals.append([cursor, start])
            labels.append("unknown")
        start = max(start, cursor)
        if end > start:
            intervals.append([start, end])
            labels.append(label)
            cursor = end

    if end_time is not None and cursor < end_time:
        intervals.append([cursor, end_time])
        labels.append("unknown")

    return np.asarray(intervals, dtype=float), labels


def song_structure_analysis(result_list):
    boundary_f1_05 = []
    boundary_f1_30 = []
    pairwise_f1 = []
    rand_scores = []
    ari_scores = []
    ref_to_est_dev = []
    est_to_ref_dev = []
    failed = 0

    for tmp in result_list:
        ref_segments = parse_song_structure_segments(tmp["correct_answer"])
        est_segments = parse_song_structure_segments(tmp["response"])

        if not ref_segments or not est_segments:
            failed += 1
            boundary_f1_05.append(0.0)
            boundary_f1_30.append(0.0)
            pairwise_f1.append(0.0)
            rand_scores.append(0.0)
            ari_scores.append(0.0)
            ref_to_est_dev.append(0.0)
            est_to_ref_dev.append(0.0)
            continue

        ref_intervals, ref_labels = segments_to_intervals_and_labels(ref_segments)
        ref_start = float(ref_intervals[0, 0])
        ref_end = float(ref_intervals[-1, 1])
        est_intervals, est_labels = segments_to_intervals_and_labels(
            est_segments,
            start_time=ref_start,
            end_time=ref_end,
        )
        ref_intervals = ref_intervals - ref_start
        est_intervals = est_intervals - ref_start

        try:
            _, _, f1_05 = mir_eval.segment.detection(
                ref_intervals,
                est_intervals,
                window=0.5,
                trim=True,
            )
            _, _, f1_30 = mir_eval.segment.detection(
                ref_intervals,
                est_intervals,
                window=3.0,
                trim=True,
            )
            _, _, pair_f1 = mir_eval.segment.pairwise(
                ref_intervals,
                ref_labels,
                est_intervals,
                est_labels,
            )
            rand_score = mir_eval.segment.rand_index(
                ref_intervals,
                ref_labels,
                est_intervals,
                est_labels,
            )
            ari_score = mir_eval.segment.ari(
                ref_intervals,
                ref_labels,
                est_intervals,
                est_labels,
            )
            ref_dev, est_dev = mir_eval.segment.deviation(
                ref_intervals,
                est_intervals,
                trim=True,
            )
        except Exception:
            failed += 1
            f1_05 = f1_30 = pair_f1 = rand_score = ari_score = ref_dev = est_dev = 0.0

        boundary_f1_05.append(f1_05)
        boundary_f1_30.append(f1_30)
        pairwise_f1.append(pair_f1)
        rand_scores.append(rand_score)
        ari_scores.append(ari_score)
        ref_to_est_dev.append(ref_dev)
        est_to_ref_dev.append(est_dev)

    print(f"Song Structure Boundary F1@0.5s: {np.mean(boundary_f1_05) * 100:.2f}")
    print(f"Song Structure Boundary F1@3.0s: {np.mean(boundary_f1_30) * 100:.2f}")
    print(f"Song Structure Pairwise F1: {np.mean(pairwise_f1) * 100:.2f}")
    print(f"Song Structure Rand Index: {np.mean(rand_scores) * 100:.2f}")
    print(f"Song Structure Adjusted Rand Index: {np.mean(ari_scores) * 100:.2f}")
    print(f"Song Structure Ref-to-Est Deviation: {np.mean(ref_to_est_dev):.4f}s")
    print(f"Song Structure Est-to-Ref Deviation: {np.mean(est_to_ref_dev):.4f}s")
    if failed:
        print(f"Song Structure Failed/empty parses: {failed}/{len(result_list)}")


genre_set = {'singersongwriter', 'instrumentalrock', 'edm', 'newage', '70s', 'metal', 'alternative', 'punkrock', 'improvisation', 'worldfusion', 'country', 'progressive', 'rap', 'darkwave', 'house', 'alternativerock', 'rocknroll', 'lounge', 'grunge', 'bluesrock', 'orchestral', 'world', 'postrock', 'instrumentalpop', 'idm', 'folk', 'drumnbass', 'club', 'contemporary', 'chanson', 'deephouse', 'rnb', 'blues', 'popfolk', 'eurodance', 'electronica', 'electropop', 'latin', 'hardrock', 'celtic', 'easylistening', 'groove', 'trance', 'dubstep', 'soul', 'jazzfusion', 'atmospheric', 'downtempo', 'techno', 'hard', 'chillout', 'classicrock', 'darkambient', 'acidjazz', 'newwave', 'breakbeat', 'ethno', 'indie', '90s', 'electronic', 'dub', 'hiphop', 'bossanova', 'choir', 'minimal', 'soundtrack', 'triphop', 'synthpop', 'medieval', 'industrial', 'pop', 'swing', '80s', 'jazz', 'symphonic', 'psychedelic', 'dance', 'ambient', 'experimental', 'fusion', 'poprock', 'reggae', 'disco', '60s', 'rock', 'classical', 'funk'}
instrument_set = {'acousticguitar', 'saxophone', 'cello', 'strings', 'bass', 'bell', 'synthesizer', 'horn', 'keyboard', 'brass', 'harmonica', 'electricguitar', 'voice', 'bongo', 'guitar', 'harp', 'viola', 'pad', 'violin', 'drummachine', 'computer', 'orchestra', 'organ', 'drums', 'doublebass', 'percussion', 'acousticbassguitar', 'clarinet', 'trombone', 'accordion', 'rhodes', 'classicalguitar', 'trumpet', 'piano', 'oboe', 'flute', 'electricpiano', 'beat', 'sampler', 'pipeorgan'}
emotion_set = {'heavy', 'powerful', 'advertising', 'funny', 'motivational', 'sad', 'sexy', 'children', 'adventure', 'trailer', 'nature', 'christmas', 'energetic', 'fun', 'uplifting', 'inspiring', 'cool', 'party', 'relaxing', 'ballad', 'melancholic', 'drama', 'sport', 'film', 'romantic', 'commercial', 'love', 'dark', 'soundscape', 'background', 'summer', 'game', 'soft', 'epic', 'travel', 'slow', 'upbeat', 'positive', 'dramatic', 'space', 'deep', 'meditative', 'retro', 'documentary', 'calm', 'happy', 'emotional', 'dream', 'holiday', 'hopeful', 'groovy', 'melodic', 'fast', 'corporate', 'action', 'movie'}

TASK_CHOICES = ["all", "MTT", "EMO_valence", "EMO_arousal", "GTZAN", "VocalSet_tech", "Nsynth_instrument", "Nsynth_pitch", "ballroom_downbeat", "gtzan_beat", "ballroom_beat", "gtzan_downbeat", "SDD", "MusicCaps", "DSing", "Guzheng_Tech", "MedleyDB", "SongFormBench", "Lyra_instrument", "Lyra_genre", "Lyra_place", "Lyra_dance", "ArabAndalusian_nawba", "ArabAndalusian_tab", "ArabAndalusian_form", "ArabAndalusian_mizan", "MTG_instrument", "MTG_genre", "GS_key", "MTG_emotion", "MTG_top50tags"]


def infer_task_from_result_path(result_path):
    result_name = os.path.basename(result_path)
    if result_name.endswith(".jsonl"):
        result_name = result_name[:-6]

    for candidate_task in sorted(TASK_CHOICES, key=len, reverse=True):
        if candidate_task == "all":
            continue
        marker = f"_{candidate_task}"
        marker_pos = result_name.rfind(marker)
        if marker_pos == -1:
            continue
        tail = result_name[marker_pos + 1:]
        if tail == candidate_task or tail.startswith(f"{candidate_task}_"):
            return candidate_task
    return None


def find_result_files(model, task):
    results_json = sorted(glob.glob(f"model/results/{model}/*.jsonl"))
    if task == "all":
        return results_json
    return [result for result in results_json if infer_task_from_result_path(result) == task]


def load_result_file(result_path):
    with open(result_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    for sample in data:
        if "response" in sample:
            sample["response"] = clean_response(sample["response"])
    return data


def evaluate_data(model, task, data):
    if task == 'GS_key':
        gmean_score = key_ensamble_score(data)
        print(f"{model}_{task} G-Mean: {gmean_score:.4f}")
    elif task == "MTT":
        tags = list(np.load("data/MTT/tags.npy"))
        value = multi_label_bert(data, tags, embed="gte")
        print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        value = multi_label_bert(data, tags, embed="bert")
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    elif task == "EMO_valence":
        r2 = cal_r2(data)
        print(f"{model}_{task} R2: {r2.cpu().numpy():.4f}")
    elif task == "EMO_arousal":
        r2 = cal_r2(data)
        print(f"{model}_{task} R2: {r2.cpu().numpy():.4f}")
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
        print(f"{model}_{task}")
        beat_tracking(data, task="downbeat_tracking")
    elif task == 'gtzan_beat':
        beat_tracking(data)
    elif task == "ballroom_beat":
        print(f"{model}_{task}")
        beat_tracking(data)
    elif task == "gtzan_downbeat":
        beat_tracking(data, task="downbeat_tracking")
    elif task == "SDD":
        music_captioning(data)
    elif task == "MusicCaps":
        music_captioning(data)
    elif task == "DSing":
        wer, cer = batch_wer_cer(data)
        print(f"{model}_{task} WER: {wer*100:.2f}")
        print(f"{model}_{task} CER: {cer*100:.2f}")
    elif task == "Guzheng_Tech":
        marco_f1, micro_f1 = calculate_frame_f1(data)
        print(f"{model}_{task} Marco F1: {marco_f1*100:.2f}")
        print(f"{model}_{task} Micro F1: {micro_f1*100:.2f}")
    elif task == "MedleyDB":
        accuracy = melody_evaluation(data)
        print(f"{model}_{task} Accuracy: {accuracy:.4f}") 
    elif task == "SongFormBench":
        print(f"{model}_{task}")
        song_structure_analysis(data)
    elif task in ("Lyra_instrument", "Lyra_genre", "Lyra_place"):
        scores = lyra_multilabel_scores(data)
        print(f"{model}_{task} labels: {scores['num_labels']}")
        print(f"{model}_{task} Micro F1: {scores['micro_f1'] * 100:.2f}")
        print(f"{model}_{task} Macro F1: {scores['macro_f1'] * 100:.2f}")
        print(f"{model}_{task} Exact Match: {scores['exact_match'] * 100:.2f}")
    elif task == "Lyra_dance":
        acc = lyra_dance_acc(data)
        print(f"{model}_{task} Acc: {acc:.4f}")
    elif task in ("ArabAndalusian_nawba", "ArabAndalusian_tab", "ArabAndalusian_form", "ArabAndalusian_mizan"):
        acc = andalusian_multiclass_acc(data)
        print(f"{model}_{task} Acc: {acc:.4f}")
    elif task == "MTG_instrument":
        tags = list(instrument_set)
        value = multi_label_bert(data, tags, embed="bert")
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        value = multi_label_bert(data, tags, embed="gte")
        print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    elif task == "MTG_genre":
        tags = list(genre_set)
        value = multi_label_bert(data, tags, embed="gte")
        print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        value = multi_label_bert(data, tags, embed="bert")
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    elif task == "MTG_emotion":
        tags = list(emotion_set)
        value = multi_label_bert(data, tags, embed="gte")
        print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        value = multi_label_bert(data, tags, embed="bert")
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    elif task == "MTG_top50tags":
        tags = ["alternative", "ambient", "atmospheric", "chillout", "classical", "dance", "downtempo", "easylistening", "electronic","experimental", "folk", "funk", "hiphop", "house", "indie", "instrumentalpop", "jazz", "lounge", "metal", "newage","orchestral", "pop", "popfolk", "poprock", "reggae", "rock", "soundtrack", 
                "techno","trance", "triphop","world", "acousticguitar", "bass", "computer", "drummachine", "drums", "electricguitar", "electricpiano", "guitar", "keyboard", "piano", "strings", "synthesizer", "violin", "voice", "emotional", "energetic", "film", "happy", "relaxing"]
        value = multi_label_bert(data, tags, embed="bert")
        print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        value = multi_label_bert(data, tags, embed="gte")
        print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
    else:
        print(model, task)
        print("Task not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="qwen2", type=str,
                        help='the model result directory under model/results/')
    parser.add_argument('--task', default="MTT", type=str,
                        choices=TASK_CHOICES,
                        help='the task to evaluate')
    args = parser.parse_args()

    results_json = find_result_files(args.model, args.task)
    if not results_json:
        raise FileNotFoundError(
            f"No result files found for model={args.model!r}, task={args.task!r}. "
            f"Expected files like model/results/{args.model}/<prefix>_<args.task>.jsonl"
        )

    for result in results_json:
        task = args.task if args.task != "all" else infer_task_from_result_path(result)
        if task is None:
            print(f"Skipping {result}: could not infer task from filename")
            continue
        print(f"Result file: {result}")
        data = load_result_file(result)
        evaluate_data(args.model, task, data)


if __name__ == "__main__":
    main()

    
    
    
    
