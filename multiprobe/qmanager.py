from sklearn.metrics.pairwise import cosine_similarity
from sqlitedict import SqliteDict
from collections import defaultdict
from sortedcontainers import SortedSet
from abc import ABC, abstractmethod
from scipy.linalg import norm
from .toolkit.levelDB import DBDictionary
from pathlib import Path

import numpy as np
import glob
import shutil



def clear_dirs(dirs: [str]):
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
        Path(d).mkdir(parents=True, exist_ok=False)


class QueryMapper(ABC): 
    def __init__(self):
        super(QueryMapper).__init__()

    @property
    @abstractmethod
    def knn_comparison_count(self):
        pass

    @property
    @abstractmethod
    def feature_map(self):
        pass

    @abstractmethod
    def load_feature_map(self, qkeys, feature_vectors):
        pass
    
    @abstractmethod
    def process_query_batch(self, query_keys, signatures: [[str]]):
        # signatures: an L * N_queries 2D list of str
        pass

    @abstractmethod
    def lookup_candidates(self, query):
        pass

    @abstractmethod
    def bucket_statistics(self):
        pass

    @abstractmethod
    def lookup_knn(self, qkey, k):
        pass

    @abstractmethod
    def reset(self):
        pass

    @staticmethod
    def pairwise_cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))  


# non-persistent query mapper;
# in-memory cache
# for small datasets only
class QueryDictMapper(QueryMapper):
    def __init__(self):
        super(QueryDictMapper).__init__()
        self.q2sig = defaultdict(set)
        self.sig2buckets = defaultdict(set)
        self._feature_map = {}
        self._knn_comparison_count = 0
    
    @property
    def feature_map(self):
        return self._feature_map

    @property
    def knn_comparison_count(self):
        ret = self._knn_comparison_count
        self._knn_comparison_count = 0
        return ret

    def load_feature_map(self, qkeys, feature_vectors):
        for k, vec in zip(qkeys, feature_vectors):
            self.feature_map[k] = vec

    def process_query_batch(self, query_keys, signatures: [[str]]):
        assert len(query_keys) == len(signatures[0]), \
            "key count must equal sig. shape axis 1"
        for qi in range(len(query_keys)):
            key = query_keys[qi]
            for l in range(len(signatures)):
                sig = signatures[l][qi]
                self.q2sig[key].add(sig)
                self.sig2buckets[sig].add(key)
    
    def lookup_candidates(self, qkey, lbd=0.70): # candidates include qkey itself
        candidates = set()
        sigs = self.q2sig[qkey]
        for sig in sigs:
            bucket = self.sig2buckets[sig]
            bucket = filter(
                lambda q: QueryMapper.pairwise_cosine_similarity(
                    self.feature_map[q], self.feature_map[qkey])>=lbd,
                bucket)
            candidates.update(bucket)
        return list(candidates)
    
    def lookup_knn(self, qkey, top_k=10):
        candidates = SortedSet(key=lambda t: (-t[1], t[0]))
        sigs = self.q2sig[qkey]
        for sig in sigs:
            bucket = self.sig2buckets[sig]
            self._knn_comparison_count += len(bucket)
            for q in bucket:
                if q == qkey:
                    continue
                sim = QueryMapper.pairwise_cosine_similarity(
                    self.feature_map[q], self.feature_map[qkey])
                candidates.add((q, sim))
        candidates = candidates[: top_k]
        return candidates

    def bucket_statistics(self):
        bucket_sizes = [len(v) for v in self.sig2buckets.values()]
        avg_size = sum(bucket_sizes) / len(bucket_sizes)
        max_size = max(bucket_sizes)
        min_size = min(bucket_sizes)
        return len(self.sig2buckets), avg_size, max_size, min_size

    def reset(self):
        self.__init__()


# persistent query mapper;
# sqlite as key-value cache
# extremely slow; for small datasets only
class QuerySQLiteMapper(QueryMapper):
    def __init__(
        self, 
        path_q2sig='./tmp/q2sig.sqldict', 
        path_sig2buckets='./tmp/sig2buckets.sqldict',
        path_feature_map='./tmp/feature_map.sqldict'
    ):
        super(QuerySQLiteMapper).__init__()
        clear_dirs([
            Path(path_q2sig).parents[0], 
            Path(path_sig2buckets).parents[0], 
            Path(path_feature_map).parents[0]])
        self.q2sig = defaultdict(set)
        self.sig2buckets = defaultdict(set)
        self.path_q2sig = path_q2sig
        self.path_sig2buckets = path_sig2buckets
        self.path_feature_map = path_feature_map
        self._knn_comparison_count = 0
        self._feature_map = SqliteDict(self.path_feature_map, journal_mode="OFF")

    @property
    def knn_comparison_count(self):
        ret = self._knn_comparison_count
        self._knn_comparison_count = 0
        return ret

    @property
    def feature_map(self):
        return self._feature_map

    def load_feature_map(self, qkeys, feature_vectors):
        for k, vec in zip(qkeys, feature_vectors):
            self.feature_map[k] = vec
        self.feature_map.commit()

    def process_query_batch(self, query_keys, signatures: [[str]]):
        assert len(query_keys) == len(signatures[0]), \
            "key count must equal sig. shape axis 1"
        for qi in range(len(query_keys)):
            key = query_keys[qi]
            for l in range(len(signatures)):
                sig = signatures[l][qi]
                self.q2sig[key].add(sig)
                self.sig2buckets[sig].add(key)
        with SqliteDict(
            self.path_q2sig, autocommit=False) as sql_q2sig:
            for k, v in self.q2sig.items():
                sql_q2sig[k] = v
            sql_q2sig.commit()
        with SqliteDict(
            self.path_sig2buckets, autocommit=False) as sql_sig2buckets:
            for k, v in self.sig2buckets.items():
                cur_bucket = sql_sig2buckets.get(k, None)
                if cur_bucket is not None:
                    self.sig2buckets[k].update(cur_bucket)
                sql_sig2buckets[k] = v
            sql_sig2buckets.commit()
        self.q2sig = defaultdict(set)
        self.sig2buckets = defaultdict(set)

    def lookup_candidates(self, qkey, lbd=0.7):
        candidates = set()
        with SqliteDict(self.path_q2sig) as sql_q2sig:
            sigs = sql_q2sig[qkey]
        fmap = self.feature_map
        with SqliteDict(self.path_sig2buckets) as sql_sig2buckets:
            for sig in sigs:
                bucket = sql_sig2buckets[sig]
                bucket = filter(
                    lambda q: QueryMapper.pairwise_cosine_similarity(
                        fmap[q], fmap[qkey]) >= lbd,
                    bucket)
                candidates.update(sql_sig2buckets[sig])
        return list(candidates)

    def lookup_knn(self, qkey, top_k=10):
        candidates = SortedSet(key=lambda t: (-t[1], t[0]))
        with SqliteDict(self.path_q2sig) as sql_q2sig:
            sigs = sql_q2sig[qkey]
        fmap = self.feature_map
        with SqliteDict(self.path_sig2buckets) as sql_sig2buckets:
            for sig in sigs:
                bucket = sql_sig2buckets[sig]
                self._knn_comparison_count += len(bucket)
                for q in bucket:
                    if q == qkey:
                        continue
                    sim = QueryMapper.pairwise_cosine_similarity(
                        fmap[q], fmap[qkey])
                    candidates.add((q, sim))
            candidates = candidates[: top_k]
        return candidates

    def bucket_statistics(self):
        with SqliteDict(self.path_sig2buckets) as sql_sig2buckets:
            bucket_sizes = [len(v) for v in sql_sig2buckets.values()]
            avg_size = sum(bucket_sizes) / len(bucket_sizes)
            max_size = max(bucket_sizes)
            min_size = min(bucket_sizes)
        return len(bucket_sizes), avg_size, max_size, min_size

    def __del__(self):
        self._feature_map.close()

    def reset(self):
        self.__init__(
            self.path_q2sig, 
            self.path_sig2buckets,
            self.path_feature_map
        )


# persistent query mapper;
# levelDB cache
# used for large datasets
class QueryLevelMapper(QueryMapper):
    def __init__(
        self, 
        path_q2sig='./tmp/q2sig', 
        path_sig2buckets='./tmp/sig2buckets',
        path_feature_map='./tmp/feature_map'
    ):
        super(QueryLevelMapper).__init__()
        clear_dirs([
            Path(path_q2sig).parents[0], 
            Path(path_sig2buckets).parents[0], 
            Path(path_feature_map).parents[0]])
        self.q2sig = defaultdict(set)
        self.sig2buckets = defaultdict(set)
        self.path_q2sig = path_q2sig
        self.path_sig2buckets = path_sig2buckets
        self.path_feature_map = path_feature_map
        self._knn_comparison_count = 0
        self._feature_map = DBDictionary(
            self.path_feature_map, read_only=False)
        self.bucket_map_index = 0

    @property
    def knn_comparison_count(self):
        ret = self._knn_comparison_count
        self._knn_comparison_count = 0
        return ret
    
    @property
    def feature_map(self):
        return self._feature_map

    def load_feature_map(self, qkeys, feature_vectors):
        self.feature_map.batch_write_list(zip(qkeys, feature_vectors))

    def process_query_batch(self, query_keys, signatures: [[str]]):
        assert len(query_keys) == len(signatures[0]), \
            "key count must equal sig. shape axis 1"
        for qi in range(len(query_keys)):
            key = query_keys[qi]
            for l in range(len(signatures)):
                sig = signatures[l][qi]
                self.q2sig[key].add(sig)
                self.sig2buckets[sig].add(key)
        with DBDictionary(self.path_q2sig) as lv_q2sig:
            lv_q2sig.batch_write_dict(self.q2sig)
        self.q2sig = defaultdict(set)

        with DBDictionary(
            f"{self.path_sig2buckets}-{self.bucket_map_index:04d}") as lv_sig2buckets:
            for k, _ in self.sig2buckets.items():
                cur_bucket = lv_sig2buckets[k]
                if cur_bucket is not None:
                    self.sig2buckets[k].update(cur_bucket)
            lv_sig2buckets.batch_write_dict(self.sig2buckets)
        self.bucket_map_index += 1
        self.sig2buckets = defaultdict(set)

    def lookup_candidates(self, qkey, lbd=0.7):
        candidates = set()
        with DBDictionary(self.path_q2sig) as lv_q2sig:
            sigs = lv_q2sig[qkey]
        fmap = self.feature_map
        bucket_maps = []
        for path in glob.glob(f"{self.path_sig2buckets}*"):
            bucket_maps.append(
                DBDictionary(path, read_only=True))
        for lv_sig2buckets in bucket_maps:
            for sig in sigs:
                bucket = lv_sig2buckets[sig]
                if bucket is None: continue
                bucket = filter(
                    lambda q: QueryMapper.pairwise_cosine_similarity(
                        fmap[q], fmap[qkey]) >= lbd,
                    bucket)
                candidates.update(lv_sig2buckets[sig])
        for lv_sig2buckets in bucket_maps:
            lv_sig2buckets.close() 
        return list(candidates)

    def lookup_knn(self, qkey, top_k=10):
        candidates = SortedSet(key=lambda t: (-t[1], t[0]))
        with DBDictionary(self.path_q2sig, read_only=True) as sql_q2sig:
            sigs = sql_q2sig[qkey]
        fmap = self.feature_map
        bucket_maps = []
        for path in glob.glob(f"{self.path_sig2buckets}*"):
            bucket_maps.append(
                DBDictionary(path, read_only=True))
        for lv_sig2buckets in bucket_maps:
            for sig in sigs:
                bucket = lv_sig2buckets[sig]
                if bucket is None: continue
                self._knn_comparison_count += len(bucket)
                for q in bucket:
                    if q == qkey:
                        continue
                    sim = QueryMapper.pairwise_cosine_similarity(
                        fmap[q], fmap[qkey])
                    candidates.add((q, sim))
        candidates = candidates[: top_k]
        for lv_sig2buckets in bucket_maps:
            lv_sig2buckets.close() 
        return candidates

    def bucket_statistics(self):
        bucket_maps = []
        bucket_sizes = []
        for path in glob.glob(f"{self.path_sig2buckets}*"):
            bucket_maps.append(
                DBDictionary(path, read_only=True))
        for lv_sig2buckets in bucket_maps:
            bucket_sizes.extend([len(v) for v in lv_sig2buckets.values()])
        for lv_sig2buckets in bucket_maps:
            lv_sig2buckets.close()
        avg_size = sum(bucket_sizes) / len(bucket_sizes)
        max_size = max(bucket_sizes)
        min_size = min(bucket_sizes)
        return len(bucket_sizes), avg_size, max_size, min_size

    def reset(self):
        self._feature_map.close()
        self.__init__(
            self.path_q2sig, 
            self.path_sig2buckets,
            self.path_feature_map
        )

    def __del__(self):
        self._feature_map.close()
