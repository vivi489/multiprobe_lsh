
from multiprobe.toolkit.data_handler import RandomDataGenerator
from multiprobe.toolkit.data_handler import QueryDataGenerator
from multiprobe.toolkit.feature_engineering import QFeatureHasher

import os
import argparse
import sqlite3
import pickle



def make_random_dataset(
    data_size, vec_norm=10.0, vec_dim=256, 
    eval_size=100, dataset_dir="./tmp/randomXXX"):
    rgen = RandomDataGenerator(data_size, vec_norm, vec_dim, eval_size, dir=dataset_dir)
    rgen.generate_points()
    if eval_size > 0:
        rgen.create_ground_truth(top_k=100)


def make_query_dataset(cursor, feature_hasher, data_size, eval_size, data_dir="./tmp/qlogXXX"):
    if not os.path.exists(data_dir):
        QueryDataGenerator.retrieve_query_db(cursor, feature_hasher, False, data_size, data_dir)
    qgen = QueryDataGenerator(data_dir)
    if eval_size > 0:
        qgen.create_ground_truth(eval_size=eval_size, top_k=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dtype", help="random/query/r/q", type=str, required=True)
    parser.add_argument("-data_dir", help="random/query/r/q", type=str, required=True)
    parser.add_argument("-data_size", help="dataset size", type=int, required=True)
    parser.add_argument("-eval_size", help="ref set size", type=int, required=True)
    parser.add_argument("-qlog_path", help="raw qlog db location", type=str, required=False)
    args = parser.parse_args()

    assert args.dtype in ["random", "query", "r", "q"], "-dtype: invalid option"
    if args.dtype.startswith('q'):
        assert args.qlog_path is not None, "raw query log database unspecified"

    if args.dtype.startswith('r'):
        make_random_dataset(args.data_size, 10.0, 256, args.eval_size, args.data_dir)
    if args.dtype.startswith('q'):
        hasher = QFeatureHasher(
            dim1=None, dim2=256, 
            catdict=pickle.load(open("./data/catdict.pickle", 'rb'))
        )
        conn = sqlite3.connect(args.qlog_path)
        cur = conn.cursor()
        make_query_dataset(cur, hasher, args.data_size, args.eval_size, args.data_dir)
        conn.close()


if __name__ == "__main__":
    # python make_datasets.py -dtype=r -data_dir=./data/random100k  -data_size=100000 -eval_size=500
    # python make_datasets.py -dtype=q -data_dir=./data/qlog10k  -data_size=10000 -eval_size=1000 -qlog_path=./data/query_logs.db
    main()
