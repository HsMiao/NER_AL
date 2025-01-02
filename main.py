import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eval import evaluate, get_results
from dataset import generate_data, select_test
from data_select import cluster, fast_vote_k, vote_k, random_select
# k=256
# test_n = None

data_path = 'data/'
result_path = 'results/'

def retrieve(k, prompt_n = 8, test_n=None, methods=['cluster', 'fast_vote', 'vote', 'random']):
    test_file = data_path + (f"test_{test_n}.csv" if test_n is not None else "test.csv")
    for method in methods:
        for flag in [0, 1]:
            df = evaluate(result_path+f"{method}_{k}.csv", test_file, prompt_n, flag, "gpt-4o-mini")
            ret = "random" if flag else "similarity"
            df['embedding'] = df['embedding'].apply(lambda x: str(x.tolist()))
            df.to_csv(result_path+f"{method}_{k}_{ret}_eval.csv", index=False)

def report(k, test_n=None, methods=['cluster', 'fast_vote', 'vote', 'random']):
    result_df = []
    for method in methods:
        for flag in [0, 1]:
            ret = "random" if flag else "similarity"
            df = pd.read_csv(result_path+f"{method}_{k}_{ret}_eval.csv")
            m = f"{method}_{k}_{ret}"
            print(m)
            df_pred = df[['id', 'text', 'tokens', 'responce']].to_dict('records')
            df_gold = df[['id', 'text', 'tokens', 'ner_tags', 'ent_list', 'ent_idx_list', 'ent_type_list']].to_dict('records')
            pres, recall, f1 = get_results(df_gold, df_pred)
            result_df.append({'method': method, 'retrieval': ret, 'precision': pres, 'recall': recall, 'f1': f1})
    result_df = pd.DataFrame(result_df)
    result_df.to_csv(result_path+f"result_{k}_{test_n}.csv", index=False)

def plot(k, test_n=None):
    df = pd.read_csv(result_path+f"result_{k}_{test_n}.csv")
    df1 = df[df['retrieval'] == 'random']
    df2 = df[df['retrieval'] == 'similarity']
    x = np.arange(len(df1))
    w = 0.3

    plt.figure(figsize=(20, 5))
    plt.bar(x-w/2, df1['precision'], w, label='random', color='#1f77b4')
    plt.bar(x+w/2, df2['precision'], w, label='similarity', color='orange')
    plt.xticks(x, df1['method'])
    plt.ylabel('precision')
    plt.ylim(0.4, 0.7)
    plt.legend()
    plt.title('Precision')
    plt.savefig(result_path+f"precision_{k}_{test_n}.png")

    plt.figure(figsize=(20, 5))
    plt.bar(x-w/2, df1['recall'], w, label='random', color='#1f77b4')
    plt.bar(x+w/2, df2['recall'], w, label='similarity', color='orange')
    plt.xticks(x, df1['method'])
    plt.ylabel('recall')
    plt.ylim(0.3, 0.7)
    plt.legend()
    plt.title('Recall')
    plt.savefig(result_path+f"recall_{k}_{test_n}.png")

    plt.figure(figsize=(20, 5))
    plt.bar(x-w/2, df1['f1'], w, label='random', color='#1f77b4')
    plt.bar(x+w/2, df2['f1'], w, label='similarity', color='orange')
    plt.xticks(x, df1['method'])
    plt.ylabel('F1')
    plt.ylim(0.4, 0.7)
    plt.legend()
    plt.title('F1')
    plt.savefig(result_path+f"f1_{k}_{test_n}.png")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test_n", type=int, default=None, help="Dataset size for testing")
    argparser.add_argument("--k", type=int, default=256, help="Number of samples to select")
    argparser.add_argument("--prompt", type=int, default=8, help="Number of prompts")
    argparser.add_argument("--random", '-r', action='store_true')
    argparser.add_argument("--cluster", '-c', action='store_true')
    argparser.add_argument("--fast_vote", '-f', action='store_true')
    argparser.add_argument("--vote", '-v', action='store_true')
    args = argparser.parse_args()
    generate_data()
    methods = []
    if args.test_n is not None:
        select_test(args.test_n)
    if args.random:
        random_select(args.k)
        methods.append('random')
    if args.cluster:
        cluster(args.k)
        methods.append('cluster')
    if args.fast_vote:
        fast_vote_k(args.k)
        methods.append('fast_vote')
    if args.vote:
        vote_k(args.k)
        methods.append('vote')
    retrieve(args.k, args.prompt, args.test_n, methods)
    report(args.k, args.test_n, methods)
    plot(args.k, args.test_n)
if __name__ == "__main__":
    main()
