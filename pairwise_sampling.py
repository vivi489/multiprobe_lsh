import numpy as np
import pandas as pd

import os
import sqlite3
import pickle
import json

from sklearn.metrics.pairwise import cosine_similarity
from multiprobe.toolkit.feature_engineering import *
from multiprobe.toolkit.levelDB import DBDictionary
from multiprobe.toolkit.data_handler import sample_range


DIR_SQL = "./data/query_logs.db"
DIR_CAT_IDX_MAP = "./data/catdict.pickle"
DIR_KV_PAIRS = "./tmp/cos_sim_pairs"
DIR_SAMPLE_PAIRS = "./tmp/sample_pairs.csv"

STEP_LEN = 2000000
STRIDE_LEN = STEP_LEN // 4
SAMPLE_BATCH_SIZE = 700
N_ITER = 10


def generate_hasher_dict(sqlite_path, dump_path):
    conn = sqlite3.connect(sqlite_path)
    cur = conn.cursor()
    qhasher = QFeatureHasher()
    init_qhasher(qhasher, cur)
    qhasher.dump_catdict(dump_path)
    conn.close()


def sample_kv_pairs(prob: int=5):
    data = []
    with DBDictionary("./tmp/cos_sim_pairs") as lvdict:
        for k, v in lvdict:
            if np.random.randint(100) < prob:
                q1, q2 = k
                if len(q1) > 1 and len(q2) > 1:
                    data.append((q1, q2, v))
    df_sample = pd.DataFrame(data, columns=[
        "query1", "query2", "cosine_similarity"]).sort_values(
            "cosine_similarity", ascending=False)
    return df_sample


def scan_sql_pairs():
    global DIR_SQL
    global DIR_CAT_IDX_MAP
    global N_ITER

    global STEP_LEN
    global STRIDE_LEN
    global SAMPLE_BATCH_SIZE

    if not os.path.exists(DIR_CAT_IDX_MAP):
        generate_hasher_dict(DIR_SQL, DIR_CAT_IDX_MAP)
    
    conn = sqlite3.connect(DIR_SQL)
    cur = conn.cursor()

   
    NUM_DB_ENTRIES = next(cur.execute("SELECT MAX(pk) FROM query_log_table"))[0]

    hasher = QFeatureHasher(
        dim1=256, dim2=256, catdict=pickle.load(open(DIR_CAT_IDX_MAP, 'rb')))
    db_cache = {}

    with DBDictionary(DIR_KV_PAIRS) as lvdict:
        for iteration in range(N_ITER):
            print("sampling iteration %d" % iteration)
            loc = 1
            while True:
                print(f"FROM {loc} TILL {loc+STEP_LEN}")
                df = sample_range(cur, sample_size=SAMPLE_BATCH_SIZE, start=loc, step_size=STEP_LEN)
                feature_vectors = []
                for _, row in df.iterrows():
                    vec = hasher.digest(
                        json.loads(row["search_results"]), 
                        json.loads(row["clicks"])
                    )
                    if vec is not None:
                        feature_vectors.append(vec)
                if len(vec) == 0:
                    continue # extremely unlikely
                Q = np.stack(feature_vectors)
                sim_mat = cosine_similarity(Q)
                del Q
                for j in range(sim_mat.shape[0]):
                    for i in range(j+1, sim_mat.shape[1]):
                        q1, q2 = None, None
                        if df["query"][j] < df["query"][i]:
                            q1, q2 = df["query"][j], df["query"][i]
                        else:
                            q1, q2 = df["query"][i], df["query"][j]
                        if sim_mat[j, i] >= 0.01:
                            db_cache[(q1, q2)] = sim_mat[j, i] 
                if loc + STEP_LEN > NUM_DB_ENTRIES:
                    break
                loc += STRIDE_LEN
                if len(db_cache) >= 500000:
                    lvdict.batch_write_dict(db_cache)
                    db_cache = {}
        if len(db_cache) > 0:
            lvdict.batch_write_dict(db_cache)
            db_cache = {}      
    conn.close()


def main():
    if not os.path.exists(DIR_KV_PAIRS):
        scan_sql_pairs()
    df = sample_kv_pairs(5)
    df.to_csv(DIR_SAMPLE_PAIRS, index=False)


if __name__ == "__main__":
    main()
