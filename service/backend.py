from sqlitedict import SqliteDict
from scipy.linalg import norm
import numpy as np


def pairwise_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def lookup(query,
           top_k=20,
           path_q2sig='./demo_data/q2sig.sqldict', 
           path_sig2buckets='./demo_data/sig2buckets.sqldict', 
           path_feature_map='./demo_data/feature_map.sqldict'):
    candidates = set()
    with SqliteDict(path_q2sig) as sql_q2sig:
        sigs = sql_q2sig[query]
    with SqliteDict(path_sig2buckets) as sql_sig2buckets:
        for sig in sigs:
            candidates.update(sql_sig2buckets[sig])
    if query in candidates:
        candidates.remove(query)
    candidates = list(candidates)
    with SqliteDict(path_feature_map, autocommit=False) as feature_map:
        similarities = np.array([pairwise_cosine_similarity(feature_map[query], feature_map[c]) for c in candidates])
    ranking = np.argsort(similarities)
    return [(candidates[i], "%0.6f"%similarities[i]) for i in reversed(ranking)][: top_k]

