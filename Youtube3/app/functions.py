import numpy as np
import polars
import sentence_transformers
import sklearn

def returnSearchResultIndexes(query, df, model, dist):
    query_embedding = model.encode(query).reshape(1, -1)
    dist_arr = dist.pairwise(df.select(df.columns[4:388]).collect(), query_embedding) + dist.pairwise(df.select(df.columns[388:]).collect(), query_embedding)
    threshold = 40
    top_k = 5
    idx_below_threshold = np.argwhere(dist_arr.flatten()<threshold).flatten()
    idx_sorted = np.argsort(dist_arr[idx_below_threshold], axis=0).flatten()
    return idx_below_threshold[idx_sorted][:top_k]