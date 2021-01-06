from sqlitedict import SqliteDict
from scipy.linalg import norm
from sortedcontainers import SortedSet
import numpy as np
import glob
import random


def pairwise_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

def pick_one_query():
    with SqliteDict("./demo_data/enum.sqldict") as db:
        index = random.randint(0, len(db)-1)
        q = db[index]
    return {"value": q}

def lookup_knn(query,
           top_k=20,
           path_q2sig='./demo_data/q2sig.sqldict', 
           path_sig2buckets='./demo_data/sig2buckets', 
           path_feature_map='./demo_data/feature_map.sqldict'):
    candidates = SortedSet(key=lambda t: (-t[1], t[0]))
    with SqliteDict(path_q2sig) as db_q2sig:
        sigs = db_q2sig[query]
    fmap = SqliteDict(path_feature_map) 
    bucket_maps = []
    for path in glob.glob(f"{path_sig2buckets}*"):
        bucket_maps.append(
            SqliteDict(path))
    for db_sig2buckets in bucket_maps:
        for sig in sigs:
            bucket = db_sig2buckets.get(sig, None)
            if bucket is None: continue
            for q in bucket:
                if q == query:
                    continue
                sim = pairwise_cosine_similarity(
                    fmap[q], fmap[query])
                candidates.add((q, sim))
    candidates = candidates[: top_k]
    fmap.close()
    for db_sig2buckets in bucket_maps:
        db_sig2buckets.close() 
    return candidates
