import json
import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter, defaultdict
from utils import test_sent, complete_prompt
from sklearn.metrics.pairwise import cosine_similarity

def similarity_retrieve(embed, df, num=8):
    matrix = np.vstack(df.embedding.values)
    # get the similarity between the input embedding and the embeddings in the file
    sim = cosine_similarity(matrix, embed.reshape(1, -1)).flatten()
    # sort the similarities
    sorted_indices = np.argsort(sim).tolist()[-num:]
    return df.iloc[sorted_indices]

def random_retrieve(embed, df, rng, num=8):
    df = df.sample(num, random_state=rng)
    matrix = np.vstack(df.embedding.values)
    # get the similarity between the input embedding and the embeddings in the file
    sim = cosine_similarity(matrix, embed.reshape(1, -1)).flatten()
    # sort df by the similarities
    df['similarity'] = sim
    df = df.sort_values('similarity')
    return df.drop(columns='similarity')

def eval_row(row, sel_df, rng, num=8, use_random=False, seed=42, model="gpt-4o-mini", together=False):
    if use_random:
        if rng is None:
            raise ValueError("Random retrieval requires a random state.")
        tmp_df = random_retrieve(row.embedding, sel_df, rng, num)
    else:
        tmp_df = similarity_retrieve(row.embedding, sel_df, num)
    sys_prompt = complete_prompt(tmp_df)
    return test_sent(sys_prompt, row.text, seed, model, together)

def evaluate(sel_file, test_file, rng=None, few_shot_num=8, use_random=False, seed=42, model="gpt-4o-mini", together=False):
    test_df = pd.read_csv(test_file)
    test_df['embedding'] = test_df['embedding'].apply(eval).apply(np.array)
    sel_df = pd.read_csv(sel_file)
    sel_df['embedding'] = sel_df['embedding'].apply(eval).apply(np.array)
    test_df[['responce', 'logprobs']] = test_df.apply(lambda row: eval_row(row, sel_df, rng, few_shot_num, use_random, model, seed=seed, together=together), axis=1).apply(pd.Series)
    return test_df
    
def tokenwise_accuracy(y_true, y_pred):
    return sum(int(yt == yp) for yt, yp in zip(y_true, y_pred)), len(y_true)

# {"named entities": [{"name": "ent_name_1", "type": "ent_type_1"}, ..., {"name": "ent_name_n", "type": "ent_type_n"}]}


def entity_index(ent_list_pred, tokens):
    sentence = ' '.join(tokens)
    ent_list_pred_indexed = []
    token_idx_start = 0
    
    for ent_pred in ent_list_pred:
        ent_pred_index = ent_pred.copy()
        ent_name = ent_pred['name']
        ent_len = len(ent_name.split())
        for token_idx in range(token_idx_start, len(tokens)):
            # if ent_name == 'adenomatous polyps':
            #     print(token_idx_start, tokens)
            #     print(' '.join(tokens[token_idx:token_idx+ent_len]))
            if ent_name == ' '.join(tokens[token_idx:token_idx+ent_len]):
                ent_pred_index['token_idx'] = (token_idx, token_idx+ent_len-1)
                token_idx_start = token_idx+ent_len
                # print(' '.join(tokens[token_idx:token_idx+ent_len]))
                break
            
        if 'token_idx' not in ent_pred_index:
            print(f"Entity '{ent_name}' not found in the sentence '{sentence}'.")
        else:
            ent_list_pred_indexed.append(ent_pred_index)
        
    return ent_list_pred_indexed

def process_miss_duplicated_entities_index_free(data_item, ent_list):
    
    sen_str = data_item['text']
    ent_names = [ent['name'] for ent in ent_list]
    ent_names_dict = defaultdict(list)
    for ent in ent_list:
        ent_names_dict[ent['name']].append(ent)
    
    ent_list_processed = []
    for name in sorted(ent_names_dict.keys(), key=lambda x: len(x.split()), reverse=True):
        freq = len(ent_names_dict[name])
        ent_per_name = ent_names_dict[name]
        ent_names_other = [other_name for other_name in ent_names if other_name != name]
        
        name_len = len(name.split())
        name_occurrence = 0
        for token_idx in range(len(sen_str.split()) - name_len + 1):
            if name == ' '.join(sen_str.split()[token_idx:token_idx+name_len]):
                name_occurrence += 1
        if name_occurrence != freq and ' '.join(ent_names).count(name) != name_occurrence:
            ent_per_name = [ent_per_name[0]] * (name_occurrence-' '.join(ent_names_other).count(name))
            
        ent_list_processed += ent_per_name
        
    return ent_list_processed

def process_miss_duplicated_entities(data_item, ent_list):
    
        sen_str = data_item['text']
        tokens = eval(data_item['tokens'])
        ent_names = [ent['name'] for ent in ent_list]
        ent_names_dict = defaultdict(list)
        for ent in ent_list:
            ent_names_dict[ent['name']].append(ent)
            
        ent_list_processed, ent_token_index_list = [], []
        ent_list_processed = ent_list.copy()
        for ent in ent_list_processed:
            ent_token_index_list += list(range(ent['token_idx'][0], ent['token_idx'][1]+1))
        
        for name in sorted(ent_names_dict.keys(), key=lambda x: len(x.split()), reverse=True):
            ent_per_name = ent_names_dict[name]
            
            if len(set([ent['type'] for ent in ent_per_name])) != 1:
                print(f"Entities: {ent_per_name}")
                print(f"Different entity types for the same entity name: {set([ent['type'] for ent in ent_per_name])}")
            ent_type_most_common = Counter([ent['type'] for ent in ent_per_name]).most_common(1)[0][0]
            
            name_len = len(name.split())
            for token_idx in range(len(tokens) - name_len + 1):
                if name == ' '.join(tokens[token_idx:token_idx+name_len]) and all([token_idx not in ent_token_index_list for token_idx in range(token_idx, token_idx+name_len)]):
                    ent_list_processed.append({'name': name, 'type': ent_type_most_common, 'token_idx': (token_idx, token_idx+name_len-1)})
                    ent_token_index_list += list(range(token_idx, token_idx+name_len))
            
        ent_list_processed = sorted(ent_list_processed, key=lambda x: x['token_idx'][0])
        
        # Make sure no over-predicted entities
        ent_name_list = [ent['name'] for ent in ent_list_processed]
        ent_name_tokens = ' '.join(ent_name_list).split()
        for token, token_count in Counter(ent_name_tokens).items():
            if token_count > tokens.count(token):
                print(f"Sentence ID: {[data_item['id']]}")
                print(tokens)
                print(ent_list)
            assert token_count <= tokens.count(token), f"Token '{token}' occurs {token_count} times in the entity names but {tokens.count(token)} times in the sentence."
            
        # Make sure no overlapping entities
        for ent in ent_list_processed:
            assert ent['name'] == ' '.join(tokens[ent['token_idx'][0]:ent['token_idx'][1]+1]), f"Entity '{ent['name']}' does not match the tokens '{tokens[ent['token_idx'][0]:ent['token_idx'][1]+1]}' in the sentence '{sen_str}'."
        
        if Counter(ent_token_index_list).most_common(1):
            assert Counter(ent_token_index_list).most_common(1)[0][1] == 1, f"Token indices {Counter(ent_token_index_list).most_common(1)[0][0]} occurs more than once in the entity list."
            
        return ent_list_processed

def get_results(gold_data, pred_path, entity_types=['Disease'], verbose=False, return_f1=False):
    """
    pred_data: list of dict
    gold_data: list of dict
    """
    if type(pred_path) == str:
        with open(pred_path, 'r') as f:
            pred_data = [json.loads(line) for line in f]
    else:
        pred_data = pred_path

    TP, FP, FN = 0, 0, 0

    for pred_item in sorted(pred_data, key=lambda x: int(str(x['id']))):
        
        gold_items = [item for item in gold_data if str(item['id']) == str(pred_item['id'])]
        assert len(gold_items) == 1
        gold_item = gold_items[0]
        gold_item['ent_list'] = eval(gold_item['ent_list'])
        gold_item['ent_type_list'] = eval(gold_item['ent_type_list'])
        gold_item['ent_idx_list'] = eval(gold_item['ent_idx_list'])
        gold_entities = [{'name': name, 'type': ent_type, 'token_idx': token_idx} for name, ent_type, token_idx in zip(gold_item['ent_list'], gold_item['ent_type_list'], gold_item['ent_idx_list'])]
        # print(gold_entities)
        sen_id = pred_item['id']
        sen_str = pred_item['text']
        sen_str_full = ' '.join(pred_item['tokens_org_full']) if 'tokens_org_full' in pred_item else sen_str
        pred_entities = json.loads(pred_item['responce'])['named entities']
        
        pred_entities = [item for item in pred_entities if item['name'] in sen_str]
        pred_entities = entity_index(pred_entities, eval(pred_item['tokens']))
        if not (pred_entities) or (pred_entities and 'token_idx' in pred_entities[0]):
            pred_entities = process_miss_duplicated_entities(pred_item, pred_entities)
        else:
            pred_entities = process_miss_duplicated_entities_index_free(pred_item, pred_entities)

        gold_entities = [each_ent for each_ent in gold_entities if each_ent['type'] in entity_types]
        pred_entities = [each_ent for each_ent in pred_entities if each_ent['type'] in entity_types]
        
        gold_entities = ['::'.join([each_ent['name'], each_ent['type'], str(each_ent['token_idx'])]) for each_ent in gold_entities]
        pred_entities = ['::'.join([each_ent['name'], each_ent['type'], str(each_ent['token_idx'])]) for each_ent in pred_entities]
        
        TP_ent = list((Counter(gold_entities) & Counter(pred_entities)).elements())
        FP_ent = list((Counter(pred_entities) - Counter(gold_entities)).elements())
        FN_ent = list((Counter(gold_entities) - Counter(pred_entities)).elements())
        
        TP += len(TP_ent)
        FP += len(FP_ent)
        FN += len(FN_ent)

        if verbose and ((Counter(pred_entities) - Counter(gold_entities)).keys() or (Counter(gold_entities) - Counter(pred_entities)).keys()):
            print(f"Sentence {sen_id}: ")
            print(f"Full sentence: {sen_str_full}")
            print(f"True Positive: {TP_ent}")
            print(f"False Positive: {FP_ent}")
            print(f"False Negative: {FN_ent}")
            # print(f"Predicted reverse: {ent_reverse}")
            print()
            # break

    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print(f"Precision: {precision*100:.1f}, Recall: {recall*100:.1f}, F1: {f1*100:.1f}\n")
    
    if return_f1:
        return f1
    else:
        return precision, recall, f1