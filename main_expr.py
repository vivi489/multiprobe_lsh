from sqlitedict import SqliteDict
from multiprobe.lsh import *
from multiprobe.qmanager import *
from multiprobe.evaluation import *
from tqdm import tqdm
from filelock import FileLock
from pathlib import Path
from timeit import default_timer
from copy import deepcopy

import numpy as np
import os
import sys
import pickle
import json
import itertools



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


def select_lsh_family(lsh_id, lsh_params: dict, weights=None):
    lsh = None
    if lsh_id == 0:
        lsh = DWiseHplaneLSH(**lsh_params, weights=weights)
    elif lsh_id == 1:
        lsh = HplaneFlipQLSH(**lsh_params)
    elif lsh_id == 2:
        lsh = CrossPolytopeLSH(**lsh_params)
    elif lsh_id == 3:
        lsh = MultiprobeCrossPolytopeLSH(**lsh_params)
    return lsh


def expand_param_dict(conf: dict):
    iter_param_lists = []
    iter_param_names = []
    for k, v in conf["params"].items():
        iter_param_names.append(k)
        if isinstance(v, list):
            iter_param_lists.append(v)
        else:
            iter_param_lists.append([v])
    ret = []
    for params in itertools.product(*iter_param_lists):
        to_add = deepcopy(conf)
        for param_name, param_value in zip(iter_param_names, params):
            to_add["params"][param_name] = param_value
        ret.append(to_add)
    return ret


def experiment(data_path: str, lsh_params: dict, lsh_id: int, mapper: QueryMapper, log_file_path="./reports.log"):
    weights = None
    lsh = None
    if lsh_id == 0:
        if not os.path.exists(os.path.join(data_path, "weights.pickle")):
            with SqliteDict(os.path.join(data_path, "data.sqldict")) as data_db:
                print("computing dimensional distributions from the dataset...")
                weights = distribution_from_data_db(data_db, lsh_params["vec_dim"])
            pickle.dump(weights, open(os.path.join(data_path, "weights.pickle"), "wb"))
        else:
            weights = pickle.load(open(os.path.join(data_path, "weights.pickle"), "rb"))
    lsh = select_lsh_family(lsh_id, lsh_params, weights=weights)
    
    with SqliteDict(os.path.join(data_path, "data.sqldict")) as data_db:
        print("processing signatures...")
        pbar = tqdm(total=len(data_db), file=sys.stdout)
        for qkeys, feature_vectors in next_batch_from_dataset(data_db, batch_size=50000):
            sigs = lsh.get_signatures(feature_vectors)
            mapper.load_feature_map(qkeys, feature_vectors)
            mapper.process_query_batch(qkeys, sigs)
            pbar.update(len(qkeys))
        pbar.close()
        n_buckets, avg_size, max_size, min_size = mapper.bucket_statistics()
        print(
            f"""n_buckets: {n_buckets}\n"""
            f"""avg_size: {avg_size}\n"""
            f"""max_size: {max_size}\n"""
            f"""min_size: {min_size}\n"""
        )
    del lsh
    # evaluations
    qkeys, ref, retrieved = [], [], []
    with SqliteDict(os.path.join(data_path, "ref.sqldict")) as ref_db:
        print("evaluating...")
        pbar = tqdm(total=len(ref_db), file=sys.stdout)
        query_time = 0.0
        for k, v in ref_db.items():
            start_time = default_timer()
            retrieved_list = [t[0] for t in mapper.lookup_knn(k, top_k=20)]
            end_time = default_timer()
            query_time += end_time - start_time
            ref_list = v[: len(retrieved_list)]
            if len(retrieved_list)>0 and len(ref_list)>0:
                qkeys.append(k)
                retrieved.append(retrieved_list)
                ref.append(ref_list)
            pbar.update(1)
        pbar.close()
    if len(qkeys) == 0:
        print("0 ref list retrieved; evaluation terminated")
        return
    evaluator = Evaluator(mapper.feature_map)
    recall = evaluator.recall(ref, retrieved)
    map_score = evaluator.map_score(ref, retrieved)
    err = evaluator.effective_err(qkeys, ref, retrieved)
    comparison_count = mapper.knn_comparison_count
    avg_query_time = query_time / len(qkeys) * 1000
    del mapper
    print("recall", recall)
    print("MAP", map_score)
    print("error ratio", err)
    print("avg query time", avg_query_time)
    print("comparison count", comparison_count)
    with FileLock("report_lock.lock"):
        print("Generating experiment report...")
        if not os.path.exists("report.tsv"):
            with open("report.tsv", "w") as f_report:
                f_report.write(
                    f"""data_path\tlsh_id\tlsh_params\trecall\tMAP\t"""
                    f"""error ratio\t#buckets\tavg_size\tmax_size\tmin_size\t"""
                    f"""comparison count\tavg query time\n""")
        with open("report.tsv", "a") as f_report:
        # work with the file as it is now locked
            f_report.write(
                f"{data_path}\t{lsh_id}\t{json.dumps(lsh_params)}\t")
            f_report.write(
                f"{recall}\t{map_score}\t{err}\t")
            f_report.write(
                f"{n_buckets}\t{avg_size}\t{max_size}\t{min_size}\t")
            f_report.write(f"{comparison_count}\t{avg_query_time}")
            f_report.write('\n')
        print("Report completed...\n")

def main(argv):
    assert len(argv) == 1, "missing configuration json file"
    configurations = json.load(open(argv[0]))
    for config_dict in configurations:
        configs = expand_param_dict(config_dict)
        for conf in configs:
            lsh_id = conf["lsh_id"]
            lsh_params = conf["params"]
            data_path = conf["data_path"]
            repeat = conf["repeat"]
            mapper_cache = conf["mapper_cache"] \
                if "mapper_cache" in conf else None
            mapper = None
            if mapper_cache is not None:
                mapper = QueryLevelMapper(
                    path_q2sig=f"{mapper_cache}/q2sig",
                    path_sig2buckets=f"{mapper_cache}/sig2buckets",
                    path_feature_map=f"{mapper_cache}/feature_map"
                )
                # mapper = QuerySQLiteMapper(
                #     path_q2sig=f"{mapper_cache}/q2sig.sqlite",
                #     path_sig2buckets=f"{mapper_cache}/sig2buckets.sqlite",
                #     path_feature_map=f"{mapper_cache}/feature_map.sqlite"
                # )
            else:
                mapper = QueryDictMapper()
            for _ in range(repeat):
                experiment(data_path, lsh_params, lsh_id, mapper)
                mapper.reset()

if __name__ == "__main__":
    main(sys.argv[1: ])
