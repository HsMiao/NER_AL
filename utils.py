import sys
import numpy as np
import pandas as pd
import os
import json
from openai import OpenAI
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

def _fast_vote_k(embeddings, select_num, k, vote_file=None):
    n = len(embeddings)
    if vote_file is not None and os.path.isfile(vote_file):
        with open(vote_file) as f:
            vote_stat = json.load(f)
    else:
        bar = tqdm(range(n),desc=f'voting')
        vote_stat = defaultdict(list)
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()[-k-1:-1]
            for idx in sorted_indices:
                if idx!=i:
                    vote_stat[idx].append(i)
            bar.update(1)
        if vote_file is not None:
            with open(vote_file,'w') as f:
                json.dump(vote_stat, f)
    votes = sorted(vote_stat.items(), key=lambda x:len(x[1]), reverse=True)
    selected_indices = []
    selected_times = defaultdict(int)
    while len(selected_indices) < select_num:
        cur_scores = defaultdict(int)
        for idx, candidates in votes:
            if idx in selected_indices:
                cur_scores[idx] = -100
                continue
            for one_support in candidates:
                if not one_support in selected_indices:
                    cur_scores[idx] += 10 ** (-selected_times[one_support])
        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        selected_indices.append(int(cur_selected_idx))
        for idx_support in vote_stat[cur_selected_idx]:
            selected_times[idx_support] += 1
    return selected_indices

system_prompt = """
Here is the JSON template for named entity recognition:
{"named entities": [{"name": "ent_name_1", "type": "ent_type_1"}, ..., {"name": "ent_name_n", "type": "ent_type_n"}]}

Please identify "Disease" entities (exact text spans), following the JSON template above, and output the JSON object. If no named entities identified, output {"named entities": []}.
"""

user_prompt = """
Input: {}
Output: 
"""

# Input: Splicing defects in the ataxia - telangiectasia gene , ATM : underlying mutations and consequences .
# Output: {"named entities": [{"name": "ataxia - telangiectasia", "type": "Disease"}]}

def get_label(row):
    dic = {"named entities": []}
    for i, ent in enumerate(eval(row['ent_list'])):
        type = eval(row['ent_type_list'])[i]
        if type == 'Disease':
            dic["named entities"].append({"name": ent, "type": type})
    return dic

def complete_prompt(demo_df):
    # iterate over the dataframe
    prompt = system_prompt
    for i, row in demo_df.iterrows():
        # add the text to the prompt
        prompt += f"Input: {row['text']}\n"
        # add the expected classification to the prompt
        prompt += f"Output: {json.dumps(get_label(row))}\n"
    return prompt

def avg_logprob(logprobs, together=False):
    avg = 0
    count = 0
    if together:
        for d in logprobs.token_logprobs:
            avg += d
            count += 1
    else:
        for d in logprobs.content:
            avg += d.logprob
            count += 1
    return avg / count

def test_sent(sys_prompt, text, seed, model="gpt-4o-mini", together=False):
    if together:
        client = OpenAI(
        api_key = os.environ.get("TOGETHER_API_KEY"),
        base_url="https://api.together.xyz/v1",
        )    
    else:
        client = OpenAI()
    # get the completion and logprobs
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt.format(text)},
        ],
        logprobs=True,
        seed=seed
    )
    # return the completion and logprob
    message = completion.choices[0].message.content
    avg_lp = avg_logprob(completion.choices[0].logprobs, together)
    return message, avg_lp

def get_responces(demo_df, test_df, seed, model="gpt-4o-mini", together=False):
    # No prompt retrieval
    sys_prompt = complete_prompt(demo_df)
    test_df[['responce', 'logprobs']] = test_df['text'].apply(lambda x: test_sent(sys_prompt, x, seed, model, together=together)).apply(pd.Series)
    return test_df
