from multiprobe.lsh import *
from multiprobe.qmanager import *
from multiprobe.evaluation import *
from multiprobe.toolkit.feature_engineering import QFeatureHasher
from pathlib import Path
from sqlitedict import SqliteDict

import numpy as np
import pandas as pd
import os
import json
import pickle
import shutil



VEC_DIM = 512
DEMO_DATA_DIR = "./service/demo_data"
DF_SRC_DIR = "./data/qlog_sample_559248_300337_409165.csv"


def distribution_from_vectors(feature_vectors, vec_dim: int):
    sums = np.array([0] * vec_dim).astype(np.float32)
    squares = np.array([0] * vec_dim).astype(np.float32)
    for vec in feature_vectors:
        sums += vec
    means = sums / len(feature_vectors)
    for vec in feature_vectors:
        squares += (vec - means)**2
    variances = np.sqrt(squares / len(feature_vectors))
    weights = list(zip(means, variances))
    return weights


def main():
    global VEC_DIM
    global DEMO_DATA_DIR
    global DF_SRC_DIR

    shutil.rmtree(DEMO_DATA_DIR, ignore_errors=True)
    Path(DEMO_DATA_DIR).mkdir(parents=True)
    df = pd.read_csv(DF_SRC_DIR).sample(5000)
    hasher = QFeatureHasher(
            dim1=256, dim2=256, catdict=pickle.load(open("./data/catdict.pickle", 'rb')))
    qkeys = []
    feature_vectors = []
    for _, row in df.iterrows():
        vec = hasher.digest(
            json.loads(row["search_results"]), 
            json.loads(row["clicks"])
        )
        if vec is not None:
            feature_vectors.append(vec)
            qkeys.append(row["query"])

    lsh = StochasticHyperplaneLSH(
        vec_dim=VEC_DIM, 
        vec_norm=10, 
        L=8, K=16, F=4, 
        weights=distribution_from_vectors(feature_vectors, VEC_DIM), 
        bucket_limit=65535)
    sigs = lsh.get_signatures(feature_vectors)
    mapper = QuerySQLiteMapper(
        path_q2sig=os.path.join(DEMO_DATA_DIR, "q2sig.sqldict"), 
        path_sig2buckets=os.path.join(DEMO_DATA_DIR, "sig2buckets.sqldict"),
        path_feature_map=os.path.join(DEMO_DATA_DIR, "feature_map.sqldict")
    )
    mapper.load_feature_map(qkeys, feature_vectors)
    del feature_vectors
    mapper.process_query_batch(qkeys, sigs)
    n_buckets, avg_size, max_size, min_size = mapper.bucket_statistics()
    print(
        f"""n_buckets: {n_buckets}\n"""
        f"""avg_size: {avg_size}\n"""
        f"""max_size: {max_size}\n"""
        f"""min_size: {min_size}"""
    )

if __name__ == "__main__":
    main()
