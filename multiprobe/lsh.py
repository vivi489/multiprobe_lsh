from abc import ABC, abstractmethod
from itertools import repeat

import numpy as np
import scipy.linalg as linalg
import time
import math
import farmhash


class LSHFamily(ABC): 
    def __init__(self):
        super(LSHFamily).__init__()

    @abstractmethod
    def reset_params(self): 
        pass

    @abstractmethod
    def digest_at_table_index(self, l, queries): 
        pass

    @abstractmethod
    def digest_all_tables(self, queries): 
        pass

    @abstractmethod
    def get_signatures(self, queries): 
        pass

    @staticmethod
    def seed():
        seed = int(time.time()*1000) % 65525
        return seed


class HplaneLSH(LSHFamily):
    # K must be even
    def __init__(self, vec_dim=512, vec_norm=10, L=16, K=64, bucket_limit=None):
        assert K%2 == 0, "K must be even"
        self.vec_dim = vec_dim
        self.vec_norm = vec_norm
        self.L = L
        self.K = K
        self.bucket_limit = bucket_limit
        self.hash_tables = [] #length L
        self.reset_params()

    # K/2 random hyperplanes 
    def _generate_pseudo_table(self):
        hyperplanes = []
        for _ in range(self.K // 2):
            hyperplane = np.random.uniform(
                -self.vec_norm, self.vec_norm, size=(self.vec_dim, ))
            hyperplane /= (linalg.norm(hyperplane) / self.vec_norm)
            hyperplanes.append(hyperplane)
        # self.vec_dim * self.K/2 matrix
        return np.vstack(hyperplanes).T

    def reset_params(self):
        self.hash_tables = []
        np.random.seed(LSHFamily.seed())
        sqrt_L = math.ceil(math.sqrt(self.L*2) + 1)
        pseudo_tables = [self._generate_pseudo_table() for _ in range(sqrt_L)]
        for j in range(sqrt_L):
            for i in range(j+1, sqrt_L):
                # self.vec_dim * self.K matrix
                cur = np.hstack([pseudo_tables[j], pseudo_tables[i]])
                cur = cur / math.sqrt(2)
                self.hash_tables.append(cur)
                if len(self.hash_tables) == self.L:
                    break
        del pseudo_tables
    
    def digest_at_table_index(self, l, queries):
        # N_queries * self.K matrix
        products = np.dot(queries, self.hash_tables[l])
        return np.apply_along_axis(lambda d: d>=0, 0, products).astype(np.int8)
        
    def digest_all_tables(self, queries):
        # a length=self.L list of N_queries * self.K matrices
        return [self.digest_at_table_index(l, queries) for l in range(self.L)]
    
    # returns a self.L * N_queries 2D list of str
    def get_signatures(self, queries) -> [[str]]:
        signatures = [] # size == self.L (table count)
        table_values = self.digest_all_tables(queries)
        # table_values: signaures across L hash tables
        for table_value in table_values:
            table_value = map(
                lambda bits: ''.join(map(str, bits)), table_value)
            if self.bucket_limit is not None:
                table_value = map(
                    lambda v: str(
                        farmhash.hash64withseed(v, 2048) % self.bucket_limit),
                    table_value
                )
            table_value = list(table_value)
            signatures.append(table_value)
        return signatures
    
    
class HplaneFlipQLSH(HplaneLSH):
    def __init__(self, vec_dim=512, vec_norm=10, L=16, K=32, F=4, bucket_limit=None):
        super().__init__(vec_dim, vec_norm, L, K, bucket_limit)
        assert F >= 1, "probing sequence length is invalid"
        assert F <= K, "flipping bits cannot exceed the hash value size K"
        self.F = F
        
    def _expand_sig(self, t):
        sign, distance_rank = t
        ret = []
        for f in range(self.F):
            sign_vec = sign.copy()
            sign_vec[distance_rank[f]] = ~sign_vec[distance_rank[f]]
            ret.append(sign_vec.astype(np.int8))
        return ret

    def digest_at_table_index(self, l, queries):
        # N_queries * self.K matrix
        products = np.dot(queries, self.hash_tables[l])
        distance_ranks = np.argsort(np.abs(products))
        signs = np.apply_along_axis(lambda d: d>=0, 0, products)
        # a length=F*N_queries list of K-d vectors
        expanded_sigs = []
        for expanded_sig in map(
            lambda t: self._expand_sig(t), zip(signs, distance_ranks)):
            expanded_sigs.extend(expanded_sig)
        # an (F*N_queries) * K matrix
        assert len(expanded_sigs) == self.F * len(queries), \
            "shape assertion failed: (F*N_queries) * K matrix"
        return np.vstack(expanded_sigs)

    # returns a (self.F*self.L) * N_queries 2D list of str
    def get_signatures(self, queries) -> [[str]]:
        # length==self.L*self.F
        # shape==(self.L*self.F) * N_queries
        signatures = [] 
        # table_values: signaures across L hash tables
        table_values = self.digest_all_tables(queries)
        # table_value: an (F*N_queries) * K matrix
        for table_value in table_values:
            table_value = map(
                lambda bits: ''.join(map(str, bits)), table_value)
            if self.bucket_limit is not None:
                table_value = map(
                    lambda v: str(
                        farmhash.hash64withseed(v, 2048) % self.bucket_limit),
                    table_value
                )
            # length==(F*N_queries)
            table_value = np.array(list(table_value))
            # shape== F * N_queries
            table_value = table_value.reshape(-1, self.F).T
            signatures.extend(table_value)
        assert len(signatures[-1]) == len(queries), "invalid shape on sigs axis 1"
        return signatures


class CrossPolytopeLSH(LSHFamily):
    def __init__(self, vec_dim=512, vec_norm=10, L=10, K=8, bucket_limit=None):
        self.vec_dim = vec_dim # must be power of 2
        self.vec_norm = vec_norm
        self.bucket_limit = bucket_limit
        self.L = L
        self.K = K
        self.H_dim = linalg.hadamard(vec_dim).astype(np.float32)
        self.reset_params()

    def _rotate(self, queries, D_123):
        X = queries
        X = np.dot(X * D_123[0], self.H_dim)
        X /= (linalg.norm(X, axis=1)[:, np.newaxis] / self.vec_norm)
        X = np.dot(X * D_123[1], self.H_dim)
        X /= (linalg.norm(X, axis=1)[:, np.newaxis] / self.vec_norm)
        X = np.dot(X * D_123[2], self.H_dim)
        X /= (linalg.norm(X, axis=1)[:, np.newaxis] / self.vec_norm)
        return X

    def reset_params(self):
        np.random.seed(LSHFamily.seed())
        self.D_123s = []
        for _ in range(self.L * self.K):
            D_123 = []
            for _ in range(3):
                D = np.random.randint(2, size=self.vec_dim).reshape(1, -1)
                D = np.apply_along_axis(
                    lambda d: d if d==1 else -1, 
                    arr=D,
                    axis=0
                ).flatten()
                D_123.append(D)
            self.D_123s.append(D_123) # D_123s length = self.L * self.K

    def _normalize_columns(self, R): # deprecated
        column_norms = linalg.norm(R, axis=0)
        return R / (column_norms / self.vec_norm)

    def _reset_params(self): # deprecated
        np.random.seed(LSHFamily.seed())
        self.rotations = []
        H = linalg.hadamard(self.vec_dim)
        for _ in range(self.L * self.K):
            DH_123 = []
            for _ in range(3):
                D = np.random.randint(2, size=self.vec_dim).reshape(1, -1)
                D = np.apply_along_axis(
                    lambda d: d if d==1 else -1, 
                    arr=D,
                    axis=0
                ).flatten()
                DH_123.append(np.dot(np.diag(D), H))
            rotation = np.dot(
                np.dot(DH_123[0], DH_123[1]),
                DH_123[2]
            )
            self.rotations.append(
                self._normalize_columns(rotation))

    # returns a N_queries * self.K matrix
    def digest_at_table_index(self, l, queries): 
        table_value_queries = []
        for k in range(self.K*l, self.K*l + self.K):
            X = self._rotate(queries, self.D_123s[k])
            indices = np.argmax(np.abs(X), axis=1)
            signs = np.apply_along_axis(
                lambda d: 1 if d>=0 else -1, 
                arr=X[np.arange(len(X)), indices].reshape(1, -1), 
                axis=0
            )
            # a size=N_queries base-1 index array
            table_value = (signs * (indices+1)).astype(np.int16)
            table_value_queries.append(table_value)
        return np.vstack(table_value_queries).T

    def digest_all_tables(self, queries): 
        # a length=self.L list of N_queries * self.K matrices
        return [self.digest_at_table_index(l, queries) for l in range(self.L)]

    # returns a self.L * N_queries 2D list of str
    def get_signatures(self, queries): 
        signatures = [] # size == self.L (table count)
        table_values = self.digest_all_tables(queries)
        # table_values: signaures across L hash tables
        for table_value in table_values:
            table_value = map(
                lambda indices: ' '.join(map(str, indices)), table_value)
            if self.bucket_limit is not None:
                table_value = map(
                    lambda v: str(
                        farmhash.hash64withseed(v, 2048) % self.bucket_limit),
                    table_value
                )
            table_value = list(table_value)
            signatures.append(table_value)
        return signatures


class MultiprobeCrossPolytopeLSH(CrossPolytopeLSH): 
    def __init__(self, vec_dim=512, vec_norm=10, L=10, M=3, bucket_limit=None):
        # vec_dim must be power of 2
        if M > vec_dim:
            assert False, "probing length exceeds the dimension scope"
        super().__init__(vec_dim=vec_dim, L=L, K=1, bucket_limit=bucket_limit)
        self.M = M
        self.reset_params()

    # returns a N_queries * self.M str matrix
    def digest_at_table_index(self, l, queries): 
        # N_queries * self.vec_dim
        X = self._rotate(queries, self.D_123s[l])
        # N_queries * self.vec_dim
        indices = np.argsort(-np.abs(X), axis=1)
        # N_queries * self.vec_dim
        signs = np.vectorize(lambda d: 1 if d>=0 else -1)(X)
        signed_sorted_indices = indices * signs
        def probe(arr: np.ndarray, probing_len):
            ret = []
            for i in range(probing_len):
                ret.append(
                    f"{i} {arr[i]}"
                )
            return np.array(ret)
        probing_sequences = np.apply_along_axis(
            lambda arr: probe(arr, self.M), 
            arr=signed_sorted_indices, 
            axis=1)
        return probing_sequences

    def digest_all_tables(self, queries): 
        # a length=self.L list of N_queries * self.M str matrices
        return [self.digest_at_table_index(l, queries) for l in range(self.L)]

    # returns a (self.M*self.L) * N_queries str matrix
    def get_signatures(self, queries): 
        probe_seq_queries = self.digest_all_tables(queries)
        ret = np.hstack(probe_seq_queries).T.tolist()
        assert len(ret) == self.M*self.L
        return ret


class DWiseHplaneLSH(HplaneFlipQLSH):
    def __init__(self, vec_dim=512, vec_norm=10, L=16, K=32, F=16, weights=None, bucket_limit=None):
        assert L <= vec_dim, "L cannot exceed dimension limit due to coordinate fix constraint"
        assert weights is not None, "precomputed normal distributions are required"
        assert vec_dim >= 2, "vector dimension has to be >= 2"
        self.means = np.array([w[0] for w in weights])
        self.variances = np.array([w[1] for w in weights])
        self.varrank = np.argsort(self.variances)
        self.flip_mask = self._flip_mask(
            self._flip_prob_from_entropy(), vec_dim)
        super().__init__(
            vec_dim=vec_dim, vec_norm=vec_norm, L=L, K=K, F=F, bucket_limit=bucket_limit)
        

    def _expand_sig(self, t):
        sign, distance_rank = t
        ret = []
        for f in range(self.F):
            sign_vec = sign.copy()
            sign_vec[distance_rank[f]] = ~sign_vec[distance_rank[f]]
            ret.append(sign_vec.astype(np.int8))
        return ret
        
    def _flip_prob_from_entropy(self):
        entropies = np.log(self.variances * math.sqrt(2 * math.pi * math.exp(1)))
        prob = entropies / entropies.max()
        return prob

    def _flip_mask(self, prob, vec_dim):
        mask = np.random.binomial(1, prob)
        mask = -mask
        for i in range(vec_dim // 4):
            if mask[self.varrank[i]] == 0:
                mask[self.varrank[i]] = 1
        for i in range(vec_dim // 4, vec_dim):
            mask[self.varrank[i]] = 1
        assert 0 not in mask, f"mask: {mask}"
        return mask

    def _generate_table_with_fixed_dim(self, l):
        hplanes = []
        for _ in range(self.K):
            r = self.varrank[l]
            fixed_coordinate = np.random.normal(self.means[r], math.sqrt(self.variances[r]))
            fixed_coordinate = min(self.vec_norm * 0.99, fixed_coordinate)
            fixed_coordinate = max(-self.vec_norm * 0.99, fixed_coordinate)

            hplane = np.random.uniform(-10, 10, size=(self.vec_dim - 1, ))
            partial_norm = math.sqrt(self.vec_norm**2 - fixed_coordinate**2)
            hplane /= linalg.norm(hplane) / partial_norm
            hplane = np.hstack([hplane[: r], [fixed_coordinate], hplane[r: ]])
            i = self.varrank[-1]
            # hplane[i] = -hplane[i]
            hplane = hplane * self.flip_mask
            hplanes.append(hplane)
        # a self.vec_dim * self.K matrix
        # row at index l is fixed by Gaussian estimation
        return np.vstack(hplanes).T

    def reset_params(self):
        np.random.seed(LSHFamily.seed())
        self.hash_tables = [
            self._generate_table_with_fixed_dim(l) for l in range(self.L)]
