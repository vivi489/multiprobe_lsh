import plyvel
import pickle

class DBDictionary:
    def __init__(self, db_path, read_only=False):
        self.read_only = read_only
        self._db = plyvel.DB(db_path, create_if_missing=not read_only)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._db.close()

    def close(self):
        self._db.close()

    def head(self, size=100):
        c = 0
        ret = []
        with self._db.iterator() as it:
            for k, v in it:
                ret.append((pickle.loads(k), pickle.loads(v)))
                c += 1
                if c >= size:
                    break
        return ret
                    
    def keys(self, decode=False):
        with self._db.iterator() as it:
            for k, _ in it:
                if decode:
                    yield pickle.loads(k)
                else:
                    yield k

    def values(self, decode=False):
        with self._db.iterator() as it:
            for _, v in it:
                if decode:
                    yield pickle.loads(v)
                else:
                    yield v

    def batch_gen(self, buffer_size):
        buffer = []
        with self._db.iterator() as it:
            for k, v in it:
                buffer.append(
                    (pickle.loads(k), pickle.loads(v)))
                if len(buffer) >= buffer_size:
                    yield buffer
                    buffer = []
        if len(buffer) >= 0:
            yield buffer
        return None
    
    def __getitem__(self, key):
        k = pickle.dumps(key)
        v = self._db.get(k)
        return pickle.loads(v) if v is not None else None
    
    def __setitem__(self, key, value):
        assert not self.read_only, "writing forbidden"
        self._db.put(
            pickle.dumps(key), pickle.dumps(value))
        
    def batch_write_dict(self, buffer: dict):
        assert not self.read_only, "writing forbidden"
        with self._db.write_batch() as wb:
            for k, v in buffer.items():
                wb.put(
                    pickle.dumps(k), pickle.dumps(v))
    
    def batch_write_list(self, buffer: list):
        assert not self.read_only, "writing forbidden"
        with self._db.write_batch() as wb:
            for k, v in buffer:
                wb.put(
                    pickle.dumps(k), pickle.dumps(v))

    def __iter__(self):
        with self._db.iterator() as it:
            for k, v in it:
                yield pickle.loads(k), pickle.loads(v)
        
    def __len__(self):
        count = 0
        with self._db.iterator() as it:
            for _ in it:
                count += 1
        return count

    def items(self):
        self.__iter__()

    def __del__(self):
        self.close()
