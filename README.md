# MULTIPROBE_LSH
Utility Package for Multiprobe LSH Families
## Dependency List
* Anaconda3 for Python 3.8
* plyvel
* sqlitedict
* Flask 1.1.2+ (service demo only)

## Dataset Creation
```shell
mkdir ./data
export DS_NAME=random10k
# number of data points
export DS_SIZE=10000
# number of reference points used as ground truth
# beware of this number:
# DS_SIZE*DS_EVAL_SIZE similarity comparisons will be computed
export DS_EVAL_SIZE=500
python make_datasets.py -dtype=r -data_dir=./data/$DS_NAME  -data_size=$DS_SIZE -eval_size=$DS_EVAL_SIZE
```

## LSH Family Benchmark
Make sure experiment configuration files in ./expr_configs are correctly set.
Suppose in ```./expr_configs/default.json```
```json
{
  "lsh_id": 0,
  "params": {
    "vec_dim": 256,
    "vec_norm": 10,
    "L": [2,3,4,5,6,7,8,9,10],
    "K": [16,14,12],
    "F": [2,3,4,5,6,7,8]
    "bucket_limit": null
  },
  "data_path": "./data/random1M",
  "mapper_cache": "./tmp/random1M",
  "repeat": 1
}
```
```mapper_cache``` is not a required field and if it absent all the signature mapping and buckets will be cached in memory. Unless you have at least 64GB local RAM, you have to specify the location for on-disk cache for 1M size datasets otherwise your OS kernel will kill your process for OOM.
Run the main experiments by
```shell
python main_expr.py ./expr_configs/default.json
```
All the hyperparameters in lists will be crosslisted so 9 * 3 * 6=162 experiment trials will be performed. Here ```lsh_id``` has three options to decide which multiprobe LSH to use. The mapping is
```
0: DWiseHplaneLSH
1: HplaneFlipQLSH
3: MultiprobeCrossPolytopeLSH
``` 

For query log data (access will be available soon), specify the configuration json file as follows.
```json
{
  "lsh_id": 0,
  "params": {
    "vec_dim": 256,
    "vec_norm": 10,
    "L": [2,3,4,5,6,7,8,9,10],
    "K": [16,14,12],
    "F": [2,3,4,5,6,7,8]
    "bucket_limit": null
  },
  "data_path": "./data/qlog1M",
  "mapper_cache": "./tmp/qlog1M",
  "repeat": 1
}
```

If reference data is not yet ready, do
```shell
export DS_EVAL_SIZE=100
export DS_NAME=qlog1M
python make_datasets.py -dtype=q -data_dir=./data/$DS_NAME -eval_size=$DS_EVAL_SIZE
```
Experiment reports will be dumped into ./report.tsv which is automatically created if not existing.
