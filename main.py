import argparse
import os
from unittest import result
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from eval import evaluate, get_results
from dataset import generate_data, select_test
from data_select import cluster, fast_vote_k, vote_k, random_select
# k=256
# test_n = None

data_path = 'data/'

def retrieve(k, prompt_n=8, test_n=None, methods=['cluster', 'fast_vote', 'vote', 'random'], flag=None, rng=None, seed=42, model="gpt-4o-mini", together=False, result_path='results/'):
    test_file = data_path + (f"test_{test_n}.csv" if test_n is not None else "test.csv")
    for method in methods:
        if flag is None:
            for flag in [0, 1]:
                ret = "random" if flag else "similarity"
                print(f"Retrieving {method} {k} {ret}")
                if os.path.exists(result_path+f"{method}_{k}_{ret}_eval.csv"):
                    continue
                df = evaluate(result_path+f"{method}_{k}.csv", test_file, prompt_n, flag, model=model, seed=seed, together=together, rng=rng)
                df['embedding'] = df['embedding'].apply(lambda x: str(x.tolist()))
                df.to_csv(result_path+f"{method}_{k}_{ret}_eval.csv", index=False)
        else:
            ret = "random" if flag else "similarity"
            print(f"Retrieving {method} {k} {ret}")
            if os.path.exists(result_path+f"{method}_{k}_{ret}_eval.csv"):
                continue
            df = evaluate(result_path+f"{method}_{k}.csv", test_file, prompt_n, flag, model=model, seed=seed, together=together, rng=rng)
            df['embedding'] = df['embedding'].apply(lambda x: str(x.tolist()))
            df.to_csv(result_path+f"{method}_{k}_{ret}_eval.csv", index=False)

def report(k, test_n=None, methods=['cluster', 'fast_vote', 'vote', 'random'], result_path='results/'):
    if test_n is None:
        test_n = 'all'
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

def plot(k, test_n=None, result_path='results/'):
    if test_n is None:
        test_n = 'all'
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
    # plt.ylim(0.4, 0.7)
    plt.legend()
    plt.title('Precision')
    plt.savefig(result_path+f"precision_{k}_{test_n}.png")

    plt.figure(figsize=(20, 5))
    plt.bar(x-w/2, df1['recall'], w, label='random', color='#1f77b4')
    plt.bar(x+w/2, df2['recall'], w, label='similarity', color='orange')
    plt.xticks(x, df1['method'])
    plt.ylabel('recall')
    # plt.ylim(0.3, 0.7)
    plt.legend()
    plt.title('Recall')
    plt.savefig(result_path+f"recall_{k}_{test_n}.png")

    plt.figure(figsize=(20, 5))
    plt.bar(x-w/2, df1['f1'], w, label='random', color='#1f77b4')
    plt.bar(x+w/2, df2['f1'], w, label='similarity', color='orange')
    plt.xticks(x, df1['method'])
    plt.ylabel('F1')
    # plt.ylim(0.4, 0.7)
    plt.legend()
    plt.title('F1')
    plt.savefig(result_path+f"f1_{k}_{test_n}.png")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test_n", type=int, default=None, help="Dataset size for testing")
    argparser.add_argument("--k", type=int, default=256, help="Number of samples to select")
    argparser.add_argument("--prompt", type=int, default=8, help="Number of prompts")
    argparser.add_argument("--together", '-t', action='store_true', help="Use together API")
    argparser.add_argument("--model", type=str, default='gpt-4o-mini', help="Model name")
    argparser.add_argument("--data", '-d', action='store_true', help="Generate data")
    argparser.add_argument("--random", '-r', action='store_true', help="Randomly select k samples")
    argparser.add_argument("--cluster", '-c', action='store_true', help="Select k samples by clustering")
    argparser.add_argument("--fast_vote", '-f', action='store_true', help="Select k samples by fast vote k")
    argparser.add_argument("--vote", '-v', action='store_true', help="Select k samples by vote k")
    argparser.add_argument("--random_retrieval", '-rr', action='store_true', help="Retrieve prompts by random retrieval")
    argparser.add_argument("--similarity_retrieval", '-sr', action='store_true', help="Retrieve prompts by similarity retrieval")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = argparser.parse_args()
    # Set random seed
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    result_path = f'results/{args.model}/'
    os.makedirs(result_path, exist_ok=True)
    if args.data or os.path.exists(data_path) == False:
        generate_data()
    methods = []
    if args.test_n is not None:
        select_test(args.test_n, rng)
    if args.random:
        print(f"Randomly select {args.k} samples")
        random_select(args.k, rng, result_path=result_path)
        methods.append('random')
    if args.cluster:
        print(f"Cluster train for {args.k} samples")
        cluster(args.k, args.seed, result_path=result_path)
        methods.append('cluster')
    if args.fast_vote:
        print(f"Fast vote {args.k} samples")
        fast_vote_k(args.k, result_path=result_path)
        methods.append('fast_vote')
    if args.vote:
        print(f"Vote {args.k} samples")
        vote_k(args.k, args.seed, args.model, args.together, result_path=result_path)
        methods.append('vote')
    if not args.random_retrieval:
        retrieve(args.k, args.prompt, args.test_n, methods, False, seed=args.seed, model=args.model, together=args.together, result_path=result_path)
    else:
        retrieve(args.k, args.prompt, args.test_n, methods, True, rng, seed=args.seed, model=args.model, together=args.together, result_path=result_path)
        if args.similarity_retrieval:
            retrieve(args.k, args.prompt, args.test_n, methods, False, seed=args.seed, model=args.model, together=args.together, result_path=result_path)
    report(args.k, args.test_n, methods, result_path)
    plot(args.k, args.test_n, result_path)
if __name__ == "__main__":
    main()
