from multiprobe.lsh import *
from multiprobe.qmanager import *
from multiprobe.evaluation import *
from pathlib import Path
from sqlitedict import SqliteDict

import numpy as np
import pandas as pd
import os
import json
import pickle
import shutil



VEC_DIM = 256
DEMO_DATA_DIR = "./service/demo_data"
SRC_DB_DIR = "./data/toydata/data.sqldict"


def distribution_from_data_db(data_db: SqliteDict, vec_dim: int):
    sums = np.array([0] * vec_dim).astype(np.float32)
    squares = np.array([0] * vec_dim).astype(np.float32)
    for _, vec in data_db.items():
        sums += vec
    means = sums / len(data_db)
    for _, vec in data_db.items():
        squares += (vec - means)**2
    variances = np.sqrt(squares / len(data_db))
    weights = list(zip(means, variances))
    return weights


def next_batch_from_dataset(data_db: SqliteDict, batch_size=20000):
    qkeys = []
    feature_vectors = []
    for q, vec in data_db.items():
        qkeys.append(q)
        feature_vectors.append(vec)
        if len(qkeys) == batch_size:
            yield qkeys, feature_vectors
            qkeys = []
            feature_vectors = []
    if len(qkeys) > 0:
        yield qkeys, feature_vectors


def main():
    global VEC_DIM
    global DEMO_DATA_DIR
    global SRC_DB_DIR

    shutil.rmtree(DEMO_DATA_DIR, ignore_errors=True)
    Path(DEMO_DATA_DIR).mkdir(parents=True)
    db_src = SqliteDict(SRC_DB_DIR, autocommit=False)
    weights = distribution_from_data_db(db_src, VEC_DIM)
    lsh = DWiseHplaneLSH(
        vec_dim=VEC_DIM, 
        vec_norm=10, 
        L=8, K=16, F=4, 
        weights=weights, 
        bucket_limit=65535)

    mapper = QuerySQLiteMapper(
        path_q2sig=os.path.join(DEMO_DATA_DIR, "q2sig.sqldict"), 
        path_sig2buckets=os.path.join(DEMO_DATA_DIR, "sig2buckets"),
        path_feature_map=os.path.join(DEMO_DATA_DIR, "feature_map.sqldict")
    )
    for qkeys, feature_vectors in next_batch_from_dataset(db_src, 10000):
        sigs = lsh.get_signatures(feature_vectors)
        mapper.load_feature_map(qkeys, feature_vectors)
        mapper.process_query_batch(qkeys, sigs)

    n_buckets, avg_size, max_size, min_size = mapper.bucket_statistics()
    print(
        f"""n_buckets: {n_buckets}\n"""
        f"""avg_size: {avg_size}\n"""
        f"""max_size: {max_size}\n"""
        f"""min_size: {min_size}"""
    )
    with SqliteDict(os.path.join(DEMO_DATA_DIR, "feature_map.sqldict")) as db_fm:
        db_enum = SqliteDict(os.path.join(DEMO_DATA_DIR, "enum.sqldict"))
        for i, k in enumerate(db_fm.keys()):
            db_enum[i] = k
        db_enum.commit()
        db_enum.close()


if __name__ == "__main__":
    main()
