import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from utils import _fast_vote_k, get_responces

data_path = 'data/'
result_path = 'results/'

def cluster(k, seed):
    df = pd.read_csv(data_path+'train.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    matrix = np.vstack(df.embedding.values)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=seed)
    kmeans.fit(matrix)
    # for each center, find the closest embeddings by cosine similarity
    closest = []
    for center in kmeans.cluster_centers_:
        closest.append(np.argmin(cosine_similarity(matrix, center.reshape(1, -1))))
    # Save the closest texts and embeddings
    df_closest = df.iloc[closest]
    df_closest['embedding'] = df_closest['embedding'].apply(lambda x: str(x.tolist()))
    df_closest.to_csv(result_path+f"cluster_{k}.csv", index=False)
    return df_closest

def fast_vote_k(k):
    df = pd.read_csv(data_path+'train.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    matrix = np.vstack(df.embedding.values)
    df_selected = df.iloc[_fast_vote_k(matrix, k, k, "vote_stat.json")]
    df_selected['embedding'] = df_selected['embedding'].apply(lambda x: str(x.tolist()))
    df_selected.to_csv(result_path+f"fast_vote_{k}.csv", index=False)
    return df_selected

def vote_k(k, seed, model="gpt-4o-mini", together=False):
    df = pd.read_csv(data_path+'train.csv')
    df['embedding'] = df['embedding'].apply(eval).apply(np.array)
    matrix = np.vstack(df.embedding.values)
    df_selected_1 = df.iloc[_fast_vote_k(matrix, k//10, k, "vote_stat.json")]
    # get the remaining rows
    df_remaining = df[~df.index.isin(df_selected_1.index)]
    df = get_responces(df_selected_1, df_remaining, seed, model, together)
    df['embedding'] = df['embedding'].apply(lambda x: str(x.tolist()))
    df.to_csv(result_path+f"responce_{k}.csv", index=False)

    # sort by logprobs and form k bins
    df['bin'] = pd.qcut(df.logprobs, k, labels=[i for i in range(k)])
    df = df[df['bin'] < k - k//10] # remove the last bins
    # for each bin, select the row with the highest logprob
    df_selected_2 = df.loc[df.groupby('bin', observed=True)['logprobs'].idxmax()].drop(columns=['bin', 'response', 'logprobs'])
    df_selected_1['embedding'] = df_selected_1['embedding'].apply(lambda x: str(x.tolist()))
    df_selected = pd.concat([df_selected_1, df_selected_2])
    df_selected.to_csv(result_path+f"vote_{k}.csv", index=False)
    return df_selected

def random_select(k, rng):
    df = pd.read_csv(data_path+'train.csv')
    df_selected = df.sample(k, random_state=rng)
    df_selected.to_csv(result_path+f"random_{k}.csv", index=False)
    return df_selected