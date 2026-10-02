import ast
import functools
import glob

import re
import argparse
import json
import os
import string
import torch

from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.meteor_score import meteor_score as meteor_scorer
from nltk.tokenize import wordpunct_tokenize
from bert_score import score

import mir_eval
from torchmetrics import R2Score
from num2words import num2words
import jiwer
import pretty_midi
from FlagEmbedding import FlagAutoModel
from torch.nn import functional as F

 
def normalise(text):
    if type(text) == list:
        text = text[0]
    return text.replace("_", "").replace("-", "").replace("#", "\u266f").replace("'", "").replace(" ", "").replace(".", "").lower()


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
    # decimals as one number: "5.26" was read as 5 and 26, and 26 dropped, giving 5
    numbers = re.findall(r'\d+(?:\.\d+)?', response)

    # the prompt asks for a number between 1 and 9; anything else (bpm, 0.8) is not an answer
    numbers = [float(i) for i in numbers if 1 <= float(i) <= 9]
    if not numbers:
        # print("No integer found in the response:", response)
        # raise ValueError("No integer found in the response."
        return -0.5
    elif len(set(numbers)) > 1:
        # eg1, around 8 and 9 -> 8, eg2. 7, becase xxx, so 7 ->7.
        print("multiple response:", response)
        # raise ValueError(f"Multiple integers found: {numbers}. Expected only one.")
    return numbers[0]  # the first number


def _tokens(text, keep_separators=False):
    tokens = re.findall(r"[a-z0-9]+|[,;.:!?/|()\[\]\n]", str(text).lower().replace("'", ""))
    return tokens if keep_separators else [t for t in tokens if t[0].isalnum()]


def _label_spans(tokens, label):
    # token spans whose concatenation spells the label, so "hip-hop", "hip hop" and
    # "hiphop" all match "hiphop", but "trillo" does not match "trill"
    # never across punctuation: "pop, rock" is not "poprock"
    target = "".join(_tokens(label))
    spans = []
    for i in range(len(tokens)):
        joined = ""
        for j in range(i, len(tokens)):
            if not tokens[j][0].isalnum():
                break
            joined += tokens[j]
            if joined in (target, target + "s", target + "es"):
                spans.append((i, j, joined == target))
            if len(joined) >= len(target) + 2:
                break
    return spans


def matched_labels(response, labels):
    """Labels mentioned in the response; a label found only inside a longer
    matched label (e.g. "trill" inside "lip_trill"), or only as the plural of
    another label (e.g. "vocal" in "vocals" when "vocals" is a label), does not count."""
    tokens = _tokens(response, keep_separators=True)
    spans = {label: _label_spans(tokens, label) for label in labels}
    found = set()
    for label, label_spans in spans.items():
        for i, j, exact in label_spans:
            shadowed = any(a <= i and j <= b and (b - a > j - i or (a, b) == (i, j) and other_exact and not exact)
                           for other, other_spans in spans.items() if other != label
                           for a, b, other_exact in other_spans)
            if not shadowed:
                found.add(label)
                break
    return found


def extract_midi_pitch(response):
    """The single MIDI number in the response, or None if there is none or several."""
    text = re.sub(r"Example \d", " ", str(response))
    text = re.sub(r"\b[A-Ga-g][#b♯♭]?-?\d+\b", " ", text)  # note names such as C4, A#3
    text = re.sub(r"\d+\.\d+", " ", text)  # decimals such as frequencies in Hz
    numbers = {int(n) for n in re.findall(r"\d+", text) if int(n) <= 127}
    return numbers.pop() if len(numbers) == 1 else None


def get_multiclass_acc(result_list):
    golds = [tmp["correct_answer"][0] if type(tmp["correct_answer"]) == list else tmp["correct_answer"]
             for tmp in result_list]
    # numeric labels (Nsynth_pitch): substring matching cannot work, "9" occurs in "69"
    if all(str(gold).strip().isdigit() for gold in golds):
        return sum(extract_midi_pitch(tmp["response"]) == int(gold)
                   for tmp, gold in zip(result_list, golds)) / len(result_list)

    answer_list = sorted(set(str(gold) for gold in golds))
    assert len(answer_list) == len(set("".join(_tokens(answer)) for answer in answer_list))
    # credit an item when the gold label, and no other label, is in the response
    count = sum(matched_labels(tmp["response"], answer_list) == {str(gold)}
                for tmp, gold in zip(result_list, golds))
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
    
    # rescale to the gold's mean and spread on this sample, so that any increasing
    # linear map of the gold (a perfect 1-9 answer) scores R2 = 1; the gold z-scores
    # come from the whole dataset and are not standardised on the test split
    a = np.std(answer_list) / std
    b = np.mean(answer_list) - a * mean
    
    response_3 = [mean if x == -0.5 else x for x in response]
    normalised_response_3 = [(a * x + b) for x in response_3]

    r2score = R2Score()
    return r2score(torch.tensor(normalised_response_3), torch.tensor(answer_list))

def multi_label_classification(result_list, answer_list): # variable should not be called type otherwise it will override the built-in function
    # Prepare answer_list as a flattened list of all possible answers
    answer_list = sorted([ans.lower().strip() for ans in answer_list])
        
    # Initialize a list to hold response vectors
    y_true = []
    y_pred = []

    for tmp in result_list:
        # exact gold tags; tags named in the response as whole words
        correct_answers = gold_tags(tmp["correct_answer"])
        predicted = matched_labels(tmp["response"], answer_list)
        
        # Create binary vectors for the true labels and predicted labels
        true_vector = [1 if answer in correct_answers else 0 for answer in answer_list]
        pred_vector = [1 if answer in predicted else 0 for answer in answer_list]
        
        y_true.append(true_vector)
        y_pred.append(pred_vector)

    return tag_auc(np.array(y_true), np.array(y_pred))


def tag_auc(y_true, y_pred):
    """Macro ROC-AUC and PR-AUC over the tags that have both positives and negatives
    in this sample; the others are undefined and made the macro average nan."""
    keep = (y_true.sum(axis=0) > 0) & (y_true.sum(axis=0) < len(y_true))
    if not keep.all():
        print(f"{(~keep).sum()} tags without both classes in this sample are left out")
    roc_auc = roc_auc_score(y_true[:, keep], y_pred[:, keep], average='macro')
    pr_auc = average_precision_score(y_true[:, keep], y_pred[:, keep], average='macro')
    return roc_auc, pr_auc


def gold_tags(correct_answer):
    """The comma-separated gold tags as a set, so that "male" is not read out of
    "female" or "vocal" out of "no vocal"."""
    if type(correct_answer) == list:
        correct_answer = ",".join(correct_answer)
    return {tag.lower().strip() for tag in str(correct_answer).split(",") if tag.strip()}


BGE_PATH = os.environ.get("CMI_BGE_PATH", "/map-vepfs/yinghao/huggingface/bge-large-en-v1.5")
GTE_PATH = os.environ.get("CMI_GTE_PATH", "/map-vepfs/yinghao/huggingface/gte-Qwen2-7B-instruct")


@functools.lru_cache(maxsize=None)
def load_embedder(embed):
    if embed == "bge":
        return FlagAutoModel.from_finetuned(
            BGE_PATH,
            query_instruction_for_retrieval="Represent this sentence for searching relevant passages: ",
            devices="cuda:0",   # if not specified, will use all available gpus or cpu when no gpu available
        )
    if embed == "gte":
        from sentence_transformers import SentenceTransformer
        gte = SentenceTransformer(GTE_PATH, trust_remote_code=True)
        gte.max_seq_length = 8192
        return gte
    return None


def multi_label_bert(result_list, answer_list, task="emotion", embed="bge"):
    # Normalize predefined answer list
    answer_list = sorted([ans.lower().strip() for ans in answer_list])

    # Initialize lists to hold response vectors
    y_true = []
    y_pred = []

    model = gte = load_embedder(embed)

    for tmp in tqdm(result_list):
        response = tmp["response"].lower().strip()
        correct_answers = gold_tags(tmp["correct_answer"])

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
    
    # Compute standard ROC-AUC and PR-AUC
    roc_auc, pr_auc = tag_auc(y_true, y_pred)
    
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
    ref_tokens, cand_tokens = [], []
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
        # BLEU over words; passing the raw strings makes NLTK score characters
        bleu4_score += sentence_bleu([ref_split], cand_split, weights=(0.0, 0.0, 0.0, 1.0))
        bleu_score += sentence_bleu([ref_split], cand_split)
        ref_tokens.append([ref_split])
        cand_tokens.append(cand_split)
        meteor_score += meteor_scorer([ref_split], cand_split)
        # break
    rouge_score, bleu_score, bleu4_score, meteor_score = rouge_score / (len(candidates)), bleu_score / (len(candidates)), bleu4_score / (len(candidates)), meteor_score / (len(candidates))
    P, R, F1 = score(candidates, mult_reference, lang="en", verbose=True)
    bert_score = R.mean().item()
    print(f"BLEU Score: {bleu_score}")
    print(f"BLEU-4 Score: {bleu4_score}")
    print(f"Corpus BLEU Score: {corpus_bleu(ref_tokens, cand_tokens)}")
    print(f"METEOR Score: {meteor_score}")
    print(f"ROUGE Score: {rouge_score}")
    print(f"BERT Score: {bert_score}")

def key_ensamble_score(result_list):
    def get_pred(tmp):
        classes = """C major, Db major, D major, Eb major, E major, F major, Gb major, G major, Ab major, A major, Bb major, B major, C minor, Db minor, D minor, Eb minor, E minor, F minor, Gb minor, G minor, Ab minor, A minor, Bb minor, B minor""".split(", ")
        tmp = ''.join([i for i in tmp.lower().replace(" ","") 
                       if not i.isdigit()])
        tmp = tmp.replace("♯", "#").replace("♭", "b").replace("-", "")
        tmp = re.sub(r"([a-g])sharp", r"\1#", tmp)
        tmp = re.sub(r"([a-g])flat", r"\1b", tmp)
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
        }
        # sharps anywhere in the response, not only as its first two characters
        tmp = re.sub(r"[cdfga]#", lambda m: map[m.group(0)], tmp)
        # the key mentioned first, not the first one in the class list
        found = [(tmp.find(class_.lower().replace(" ", "")), class_) for class_ in classes]
        found = [(pos, class_) for pos, class_ in found if pos >= 0]
        return min(found)[1] if found else None
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
        # one unparseable token (e.g. "[0.5" or "1.0]") used to zero the whole item
        for i in re.split(r"[,;\s\[\]()]+", beats):
            match = re.fullmatch(r"(?:(\d+):)?(\d+(?:\.\d+)?)(?:s+|sec|secs|seconds)?", i.strip())
            if match:
                tmp.append(int(match.group(1) or 0) * 60 + float(match.group(2)))
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
    input_string = input_string.strip()
    if len(input_string) < 2:
        return None, None
        # raise ValueError("Empty input string")
    input_string = input_string[:-1] if input_string[-2]=="]" else input_string  #punctuation after "[]"
    # literal_eval: never execute model output
    try:
        midi_sequence = ast.literal_eval(input_string)
    except (SyntaxError, ValueError, TypeError, MemoryError, RecursionError) as e:
        if 'unterminated string literal' in str(e):
            last_paren = input_string.rfind(')')
            start_paren = input_string.find('[', 0, last_paren)
            if start_paren == -1:
                return None, None
            fixed_string = input_string[start_paren:last_paren] + ")]"
            try:
                midi_sequence = ast.literal_eval(fixed_string)
            except Exception:
                return None, None
        elif "'[' was never closed" in str(e):
            try:
                midi_sequence = ast.literal_eval(input_string + "]")
            except Exception:
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
        if isinstance(item[0], (int, float)):
            continue
        # exception:2:43, '0.00'
        time_str = str(item[0]).strip().strip("\'\"")
        midi_sequence[idx] = (time_str, item[1])
        if ":" in time_str:
            try:
                midi_sequence[idx] = (float(time_str.split(":")[1]) + float(time_str.split(":")[0]) * 60, item[1])
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
        if isinstance(note, str):  # If it's a note name, convert it
            note = note.strip()
            try:
                note = float(note)
            except ValueError:
                try:
                    note = pretty_midi.note_name_to_number(note)
                except:  # '' or a string with Chinese characters which is not a midi name
                    note = 0
        if not isinstance(note, (int, float)):
            note = 0
        # MIDI 0 marks unvoiced frames in the gold annotation: 0 Hz, not
        # note_number_to_hz(0) = 8.18 Hz, which mir_eval would count as voiced
        midi_array[i, 1] = pretty_midi.note_number_to_hz(note) if note > 0 else 0.0

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
        response_time, response_freq = process_midi_sequence(response)
        correct_time, correct_freq = process_midi_sequence(correct_answers)
        
        if response_time is None or correct_time is None:
            overall_accuracy.append(0)
            continue
        
        try:
            overall_accuracy.append(
                mir_eval.melody.evaluate(correct_time, correct_freq, 
                                        response_time, response_freq)['Overall Accuracy']
            )
        except Exception:
            print(tmp)
            overall_accuracy.append(0)  # keep failures in the denominator

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
    # the whole list: find(')') kept only the first event of every answer, gold included
    start, end = events.find('['), events.rfind(')')
    if start == -1 or start >= end:
        events = [('0','10','No Tech')]
    else:
        # should only have one "]" if following instruction
        try:
            events_string = events[start:end ] + ")]"
            if events_string.startswith("[\"["):
                events_string = events_string[2:]
            events = ast.literal_eval(events_string)
        except Exception:
            print(events)
            events = []
    if not isinstance(events, (list, tuple)):
        events = []
    parsed = []
    for e in events:
        try:
            if isinstance(e, dict):
                e = (e['start'], e['end'], e['technique'])
            parsed.append((float(e[0]), float(e[1]), normalise(str(e[2]))))
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    # times are absolute; a clip covers [10k, 10k + 10). Keep each event's duration
    # instead of wrapping its end, which dropped events running past the clip end.
    events = []
    for event_start, event_end, label in parsed:
        if label not in CLASSES or event_end <= event_start:
            continue  # "No Tech", or a technique outside the list
        clip_start = event_start % 10
        events.append((clip_start, min(clip_start + event_end - event_start, 10), label))

    max_time = 10 #max(float(event[1]) for event in events) if events else 0
    num_frames = int(np.ceil(max_time * sr))
    frame_labels = np.zeros((num_frames, len(CLASSES)))
    for event in events:
        start_frame = int(float(event[0]) * sr)
        end_frame = int(float(event[1]) * sr)
        label_idx = CLASSES.index(event[2])
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
        if tp_per_class[i] + fp_per_class[i] + fn_per_class[i] == 0:
            continue  # absent from both gold and prediction (no Plucks in the test split): F1 undefined
        precision = tp_per_class[i] / (tp_per_class[i] + fp_per_class[i]) if (tp_per_class[i] + fp_per_class[i]) > 0 else 0
        recall = tp_per_class[i] / (tp_per_class[i] + fn_per_class[i]) if (tp_per_class[i] + fn_per_class[i]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_f1.append(f1)
    
    macro_f1 = np.mean(class_f1)

    return micro_f1, macro_f1


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
    results_json = sorted(glob.glob(f"model/results/{model}/{model}_*.jsonl"))
    if task != "all":
        # exact file name: a substring test can pick another task's file
        results_json = [result for result in results_json if os.path.basename(result) == f"{model}_{task}.jsonl"]
    if not results_json:
        raise SystemExit(f"no results for --model {model} --task {task} in model/results/{model}/")
    # --task all used to score only the first file
    for result in results_json:
        task = os.path.basename(result)[len(model)+1:-6]
        # model/infer.py writes one JSON list despite the .jsonl suffix; accept JSON lines too
        with open(result, "r") as f:
            text = f.read()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = [json.loads(line) for line in text.splitlines() if line.strip()]

        print("sample", data[0])
        if task == 'GS_key':
            gmean_score = key_ensamble_score(data)
            print(f"{model}_{task} G-Mean: {gmean_score:.4f}")
        elif task == "MTT":
            tags = list(np.load("data/MTT/tags.npy"))
            # roc_auc, pr_auc = multi_label_classification(data, tags)
            # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
            # value = multi_label_bert(data, tags)
            # print(f"{model}_{task} BGE\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
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
            micro_f1, marco_f1 = calculate_frame_f1(data)  # returns (micro, macro)
            print(f"{model}_{task} Marco F1: {marco_f1*100:.2f}")
            print(f"{model}_{task} Micro F1: {micro_f1*100:.2f}")
        elif task == "MedleyDB":
            accuracy = melody_evaluation(data)
            print(f"{model}_{task} Accuracy: {accuracy:.4f}") 
        elif task == "MTG_instrument":
            tags = list(instrument_set)
            # roc_auc, pr_auc = multi_label_classification(data, tags)
            # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
            # value = multi_label_bert(data, tags)
            # print(f"{model}_{task} BGE\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            value = multi_label_bert(data, tags, embed="bert")
            print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            value = multi_label_bert(data, tags, embed="gte")
            print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            # tags = ["accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass", "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", "drums", "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice"]
        elif task == "MTG_genre":
            tags = list(genre_set)
            # roc_auc, pr_auc = multi_label_classification(data, tags)
            # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
            # value = multi_label_bert(data, tags)
            # print(f"{model}_{task} BGE\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            value = multi_label_bert(data, tags, embed="gte")
            print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            value = multi_label_bert(data, tags, embed="bert")
            print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            # tags = ["60s", "70s", "80s", "90s", "acidjazz", "alternative", "alternativerock", "ambient", "atmospheric", "blues", "bluesrock", "bossanova", "breakbeat", "celtic", "chanson", "chillout", "choir", "classical", "classicrock", "club", "contemporary", "country", "dance", "darkambient", "darkwave", "deephouse", "disco", "downtempo", "drumnbass", "dub", "dubstep", "easylistening", "edm", "electronic", "electronica", "electropop", "ethno", "eurodance", "experimental", "folk", "funk", "fusion", "groove", "grunge", "hard", "hardrock", "hiphop", "house", "idm", "improvisation", "indie", "industrial", "instrumentalpop", "instrumentalrock", "jazz", "jazzfusion", "latin", "lounge", "medieval", "metal", "minimal", "newage", "newwave", "orchestral", "pop", "popfolk", "poprock", "postrock", "progressive", "psychedelic", "punkrock", "rap", "reggae", "rnb", "rock", "rocknroll", "singersongwriter", "soul", "soundtrack", "swing", "symphonic", "synthpop", "techno", "trance", "triphop", "world", "worldfusion"]
        elif task == "MTG_emotion":
            tags = list(emotion_set)
            # roc_auc, pr_auc = multi_label_classification(data, tags)
            # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
            value = multi_label_bert(data, tags, embed="gte")
            print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            value = multi_label_bert(data, tags, embed="bert")
            print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            # value = multi_label_bert(data, tags)
            # print(f"{model}_{task} BGE\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            # tags = ["action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas", "commercial", "cool", "corporate", "dark", "deep", "documentary", "drama", "dramatic", "dream", "emotional", "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy", "happy", "heavy", "holiday", "hopeful", "inspiring", "love", "meditative", "melancholic", "melodic", "motivational", "movie", "nature", "party", "positive", "powerful", "relaxing", "retro", "romantic", "sad", "sexy", "slow", "soft", "soundscape", "space", "sport", "summer", "trailer", "travel", "upbeat", "uplifting"]
        elif task == "MTG_top50tags":
            tags = ["alternative", "ambient", "atmospheric", "chillout", "classical", "dance", "downtempo", "easylistening", "electronic","experimental", "folk", "funk", "hiphop", "house", "indie", "instrumentalpop", "jazz", "lounge", "metal", "newage","orchestral", "pop", "popfolk", "poprock", "reggae", "rock", "soundtrack", 
                    "techno","trance", "triphop","world", "acousticguitar", "bass", "computer", "drummachine", "drums", "electricguitar", "electricpiano", "guitar", "keyboard", "piano", "strings", "synthesizer", "violin", "voice", "emotional", "energetic", "film", "happy", "relaxing"]
            # roc_auc, pr_auc = multi_label_classification(data, tags)
            # print(f"{model}_{task} Accurate\n ROC-AUC: {roc_auc:.4f}\n PR-AUC: {pr_auc:.4f}")
            # value = multi_label_bert(data, tags)
            # print(f"{model}_{task} BGE\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            value = multi_label_bert(data, tags, embed="bert")
            print(f"{model}_{task} BERT\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
            value = multi_label_bert(data, tags, embed="gte")
            print(f"{model}_{task} GTE-Qwen\n ROC-AUC: {value['ROC-AUC']:.4f}\n PR-AUC: {value['PR-AUC']:.4f}")
        else:
            print(model, task)
            print("Task not found")

    
    
    
    
