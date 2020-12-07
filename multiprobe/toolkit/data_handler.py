from sqlitedict import SqliteDict
from scipy.linalg import norm
from pathlib import Path
from tqdm import tqdm
from .feature_engineering import QFeatureHasher

import numpy as np
import pandas as pd
import sqlite3
import json
import os
import sys



def sample_range(cursor, sample_size, start, step_size):
    idx_list = np.random.choice(range(start, start + step_size), sample_size, replace=False)
    idx_list = ','.join(map(str, idx_list))
    data = []
    for row in cursor.execute(f"""SELECT * FROM query_log_table WHERE pk in ({idx_list})"""):
        data.append(row)
    return pd.DataFrame(data, columns=["pk", "query", "search_results", "clicks"])


# scope \in ["all", "clicks", "search_results"]
def sample_categories(cursor, cat_ids: [str], scope="all"):
    data = []
    for row in cursor.execute(f"""SELECT * FROM query_log_table"""):
        _, _, search_results, clicks = row
        if scope == "all" or scope == "search_results":
            search_results = json.loads(search_results)
            for sr in search_results:
                sr_ = sr.split('/')
                if any(map(lambda d: d in sr_, cat_ids)):
                    data.append(row)
        if scope == "all" or scope == "clicks":
            clicks = json.loads(clicks)
            for sr in clicks.keys():
                sr_ = sr.split('/')
                if any(map(lambda d: d in sr_, cat_ids)):
                    data.append(row)
    return pd.DataFrame(data, columns=["pk", "query", "search_results", "clicks"])


def pairwise_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))


class RandomDataGenerator:
    def __init__(self, N, vec_norm, vec_dim, N_eval, dir):
        self.N_sample = N
        self.dataset_path = dir
        self.vec_norm = vec_norm
        self.vec_dim = vec_dim
        self.dir = dir
        self.N_eval = N_eval
        Path(self.dir).mkdir(parents=True, exist_ok=True)

    def generate_points(self, cache_size=4096):
        N_remaining = self.N_sample
        db = SqliteDict(os.path.join(self.dir, "data.sqldict"), autocommit=False)
        primary_key = 1
        print("making up data set...")
        pbar = tqdm(total=N_remaining, file=sys.stdout)
        while N_remaining > 0:
            vectors = []
            for _ in range(min(N_remaining, cache_size)):
                x = np.random.uniform(
                    -self.vec_norm, self.vec_norm, size=(self.vec_dim, )
                )
                x = x / (norm(x) / self.vec_norm)
                vectors.append(x)
            for vec in vectors:
                db[primary_key] = vec
                primary_key += 1
            db.commit()
            pbar.update(min(N_remaining, cache_size))
            N_remaining -= min(N_remaining, cache_size)
        pbar.close()
        assert N_remaining == 0, "R_remaining==0 expected"
    
    def create_ground_truth(self, top_k=100):
        eval_indices = np.random.choice(
            list(range(1, self.N_sample+1)), self.N_eval, replace=False)
        knn_truth = []
        print("computing global similarities...")
        pbar = tqdm(total=self.N_eval, file=sys.stdout)
        with SqliteDict(os.path.join(self.dir, "data.sqldict")) as data_db:
            for index in map(str, eval_indices):
                cur_vector = data_db[index]
                sim_list = []
                for i, v in data_db.items():
                    if i == index:
                        continue
                    sim_list.append(
                        (i, pairwise_cosine_similarity(cur_vector, v))
                    )
                sim_list.sort(key=lambda t: (-t[1], t[0]))
                sim_list = [t[0] for t in sim_list[: top_k]]
                knn_truth.append((index, sim_list))
                pbar.update(1)
        pbar.close()
        print("committing...")
        with SqliteDict(os.path.join(self.dir, "ref.sqldict"), autocommit=False) as ref_db:
            for k, sim_list in knn_truth:
                ref_db[k] = sim_list
            ref_db.commit()



class QueryDataGenerator:
    def __init__(self, dir):
        assert os.path.exists(dir), "non-existent dataset dir"
        self.dir = dir

    @staticmethod
    def retrieve_query_db(cursor, feature_hasher, data_size, data_dir):
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        qlog_table_size = None
        for row in cursor.execute("SELECT COUNT(*) FROM query_log_table"):
            qlog_table_size = row[0]
        with SqliteDict(os.path.join(data_dir, "data.sqldict"), autocommit=False) as data_db:
            while len(data_db) < data_size:
                remaining = data_size - len(data_db)
                batch_size = min(remaining, 50000)
                df = sample_range(cursor, batch_size, 1, qlog_table_size)
                qkeys = []
                feature_vectors = []
                for _, row in df.iterrows():
                    vec = feature_hasher.digest(
                        json.loads(row["search_results"]), 
                        json.loads(row["clicks"])
                    )
                    if vec is not None:
                        feature_vectors.append(vec)
                        qkeys.append(row["query"])
                for q, v in zip(qkeys, feature_vectors):
                    data_db[q] = v
                data_db.commit()
        
    def create_ground_truth(self, eval_size, top_k=100):
        dataset_keys = []
        with SqliteDict(os.path.join(self.dir, "data.sqldict")) as data_db:
            for k in data_db.keys():
                dataset_keys.append(k)
        eval_keys = np.random.choice(dataset_keys, eval_size, replace=False)
        knn_truth = []
        print("computing global similarities...")
        pbar = tqdm(total=eval_size, file=sys.stdout)
        with SqliteDict(os.path.join(self.dir, "data.sqldict")) as data_db:
            for key in eval_keys:
                cur_vector = data_db[key]
                sim_list = []
                for k, v in data_db.items():
                    if k == key:
                        continue
                    sim_list.append(
                        (k, pairwise_cosine_similarity(cur_vector, v))
                    )
                sim_list.sort(key=lambda t: (-t[1], t[0]))
                sim_list = [t[0] for t in sim_list[: top_k]]
                knn_truth.append((key, sim_list))
                pbar.update(1)
        pbar.close()
        print("committing...")
        with SqliteDict(os.path.join(self.dir, "ref.sqldict"), autocommit=False) as ref_db:
            for k, sim_list in knn_truth:
                ref_db[k] = sim_list
            ref_db.commit()
