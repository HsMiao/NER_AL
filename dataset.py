import os
import pandas as pd
from collections import defaultdict, Counter
from datasets import load_dataset
from openai import OpenAI

label2id = ["O", "B-Disease", "I-Disease"]
id2label = {0: "O", 1: "B-Disease", 2: "I-Disease"}

data_path = 'data/'

def index_ent_in_prediction(word_list, tag_list):
    ent_queue, ent_idx_queue, ent_type_queue = [], [], []
    ent_list, ent_idx_list, ent_type_list = [], [], []

    for word_idx in range(len(word_list)):

        if 'B-' in tag_list[word_idx]:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []
            ent_queue.append(word_list[word_idx])
            ent_idx_queue.append(word_idx)
            ent_type_queue.append(tag_list[word_idx][2:])

        if 'I-' in tag_list[word_idx]:
            if word_idx == 0 or (word_idx > 0 and tag_list[word_idx][2:] == tag_list[word_idx - 1][2:]):
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])
            else:
                if ent_queue:

                    if len(set(ent_type_queue)) != 1:
                        print(ent_queue)
                        print(ent_idx_queue)
                        print(ent_type_queue)
                        print(Counter(ent_type_queue).most_common())
                        print()
                    else:
                        ent_list.append(' '.join(ent_queue).strip())
                        ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                        assert len(set(ent_type_queue)) == 1
                        ent_type_list.append(ent_type_queue[0])

                ent_queue, ent_idx_queue, ent_type_queue = [], [], []
                ent_queue.append(word_list[word_idx])
                ent_idx_queue.append(word_idx)
                ent_type_queue.append(tag_list[word_idx][2:])

        if 'O' == tag_list[word_idx] or word_idx == len(word_list) - 1:
            if ent_queue:

                if len(set(ent_type_queue)) != 1:
                    print(ent_queue)
                    print(ent_idx_queue)
                    print(ent_type_queue)
                    print(Counter(ent_type_queue).most_common())
                    print()

                else:
                    ent_list.append(' '.join(ent_queue).strip())
                    ent_idx_list.append((ent_idx_queue[0], ent_idx_queue[-1]))

                    assert len(set(ent_type_queue)) == 1
                    ent_type_list.append(ent_type_queue[0])

            ent_queue, ent_idx_queue, ent_type_queue = [], [], []

    return ent_list, ent_idx_list, ent_type_list

def generate_ent_tagged_text(tokens, tags):
    assert len(tokens) == len(tags)

    ent_list, ent_idx_list, ent_type_list = index_ent_in_prediction(tokens, tags)

    ent_idx_last = 0
    tokens_processed = []
    for ent_idx, ent, ent_type in zip(ent_idx_list, ent_list, ent_type_list):
        ent_idx_start, ent_idx_end = ent_idx
        assert ent == ' '.join(tokens[ent_idx_start:ent_idx_end + 1])

        if ent_idx_start > ent_idx_last:
            tokens_processed.append(' '.join(tokens[ent_idx_last:ent_idx_start]))

        tokens_processed.append(f'<{ent_type}>' + ' '.join(tokens[ent_idx_start:ent_idx_end + 1]) + f'</{ent_type}>')
        ent_idx_last = ent_idx_end + 1

    if ent_idx_last < len(tokens):
        tokens_processed.append(' '.join(tokens[ent_idx_last:]))

    tagged_sentence = ' '.join(tokens_processed)
    
    return ent_list, ent_idx_list, ent_type_list, tagged_sentence

def process_data(data):
    
    data_processed = []
    for sen_item in data:
        id = sen_item['id']
        tokens = sen_item['tokens']
        ner_tags = [id2label[tag] for tag in sen_item['ner_tags']]
        ent_list, ent_idx_list, ent_type_list, tagged_sentence = generate_ent_tagged_text(tokens, ner_tags)
        
        sentence_dict = {
            'id': id,
            'tokens': tokens,
            'ner_tags': ner_tags,
            'ent_list': ent_list,
            'ent_idx_list': ent_idx_list,
            'ent_type_list': ent_type_list
        }
        
        data_processed.append(sentence_dict)
        
    print(f"There are {len(data_processed)} examples in the processed data.")
    data_processed = pd.DataFrame(data_processed)
    data_processed = data_processed[data_processed['tokens'].apply(len) > 0]
    data_processed['text'] = data_processed['tokens'].apply(lambda x: ' '.join(x))
    client = OpenAI()

    def get_embedding(text, model="text-embedding-3-small"):
        return client.embeddings.create(input = [text], model=model).data[0].embedding

    data_processed['embedding'] = data_processed['text'].apply(get_embedding).apply(str)
    return data_processed

def generate_data():
    raw_datasets = load_dataset("ncbi_disease.py", cache_dir="ncbi_disease")
    os.makedirs(data_path, exist_ok=True)
    train_data_raw = raw_datasets['train']
    dev_data_raw = raw_datasets['validation']
    test_data_raw = raw_datasets['test']
    train_data_processed = process_data(train_data_raw)
    dev_data_processed = process_data(dev_data_raw)
    test_data_processed = process_data(test_data_raw)
    train_data_processed.to_csv(data_path+'train_data.csv', index=False)
    dev_data_processed.to_csv(data_path+'dev_data.csv', index=False)
    test_data_processed.to_csv(data_path+'test_data.csv', index=False)

def select_test(num):
    test_data = pd.read_csv(data_path+'test_data.csv')
    test_data_selected = test_data.sample(n=num)
    test_data_selected.to_csv(data_path+f'test_data_{num}.csv', index=False)