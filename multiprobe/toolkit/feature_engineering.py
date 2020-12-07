from collections import defaultdict
import numpy as np
import sqlite3
import json
import farmhash
import pickle


class QFeatureHasher:
    def __init__(self, dim1=256, dim2=256, catdict=None):
        self.dim_search = dim1
        self.dim_clicks = dim2
        self.catid2index = {} if catdict is None else catdict
        self._catlvl2catid = defaultdict(set)
    
    # stateful methods - to be called multiple times 
    def load_cat_ids(self, search_results: [[str]], clicks: [{str: int}]):
        for search_list in search_results:
            for path in search_list:
                cat_ids = path.split('/')[1: ]
                for i, d in enumerate(cat_ids):
                    self._catlvl2catid[i+1].add(d)
        for click in clicks:
            for path in click.keys():
                cat_ids = path.split('/')[1: ]
                for i, d in enumerate(cat_ids):
                    self._catlvl2catid[i+1].add(d)
        
    def dump_catdict(self, path):
        levels = list(self._catlvl2catid.keys())
        levels.sort(reverse=True)
        for lvl in levels:
            for d in self._catlvl2catid[lvl]:
                self.catid2index[d] = len(self.catid2index)
        pickle.dump(self.catid2index, open(path, 'wb'))
        
    def digest(self, search_result: [str], click: {str: int}, norm=10.0) -> np.array:
        vec1 = [0.0] * self.dim_search
        vec2 = [0.0] * self.dim_clicks
        for path in search_result:
            if '/' not in path:
                return None
            cat_ids = path.split('/')[1: ]
            for i, d in enumerate(cat_ids):
                lv = i + 1
                index = self.catid2index[d] % self.dim_search
                value = 1.0 if farmhash.hash64withseed(d, 4321) % 2 == 0 else -1.0
                vec1[index] += value / lv
        for path, count in click.items():
            if '/' not in path:
                return None
            cat_ids = path.split('/')[1: ]
            for i, d in enumerate(cat_ids):
                lv = i + 1
                index = self.catid2index[d] % self.dim_clicks
                value = 1.0 if farmhash.hash64withseed(d, 4321) % 2 == 0 else -1.0
                vec2[index] += value / lv * count
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        # vec1 = vec1 / (np.linalg.norm(vec1) / norm)    
        # vec2 = vec2 / (np.linalg.norm(vec2) / norm)
        ret = np.concatenate((vec1, vec2))
        ret = ret / (np.linalg.norm(ret) / norm)
        if np.isnan(ret).any():
            return None
        return ret


def init_qhasher(qhasher, cursor, batch_size=500000):
    search_results, clicks = [], []
    """expected query log table schema
    CREATE TABLE query_log_table
    (
        pk INTEGER PRIMARY KEY AUTOINCREMENT,
        query TEXT NOT NULL, 
        search_results TEXT NOT NULL, 
        clicks TEXT NOT NULL
    )
    """
    for row in cursor.execute(f"""SELECT search_results, clicks FROM query_log_table"""):
        search_result, click = row
        search_results.append(json.loads(search_result))
        clicks.append(json.loads(click))
        if len(clicks) >= 500000:
            qhasher.load_cat_ids(search_results, clicks)
            search_results, clicks = [], []
    if len(clicks) >= 0:
        qhasher.load_cat_ids(search_results, clicks)
