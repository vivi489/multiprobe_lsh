from scipy.linalg import norm
import numpy as np


class Evaluator:
    def __init__(self, feature_map=None):
        self.feature_map = feature_map

    def recall(self, ref: [[str]], retrieved: [[str]]):
        assert len(ref) == len(retrieved), \
            "reference and retrieved lists count not equal"
        assert len(ref)>0 and len(retrieved)>0, "empty ref/retr lists"
        numerator = 0
        denominator = 0
        for ref_list, retrieved_list in zip(ref, retrieved):
            assert len(ref_list) == len(retrieved_list), \
                "ref list length is inconsistent with the retrieved list"
            numerator += len(
                set(ref_list) & set(retrieved_list))
            denominator += len(ref_list)
        return numerator / denominator

    def _mean_precision(self, ref_list: [str], retrieved_list: [str]):
        cutoff_precisions = []
        for size in range(1, len(ref_list) + 1):
            cutoff_precisions.append(
                len(
                    set(ref_list[: size]) & set(retrieved_list[: size])
                ) / size
            )
        is_relevant = []
        for q in retrieved_list:
            is_relevant.append(1 if q in ref_list else 0)
        return np.dot(cutoff_precisions, is_relevant) / len(ref_list)

    def map_score(self, ref: [[str]], retrieved: [[str]]):
        assert len(ref) == len(retrieved), \
            "reference and retrieved lists count not equal"
        assert len(ref)>0 and len(retrieved)>0, "empty ref/retr lists"
        mprecisions = []
        for ref_list, retrieved_list in zip(ref, retrieved):
            assert len(ref_list) == len(retrieved_list), \
                "ref list length is consistent with the retrieved list"
            mprecisions.append(
                self._mean_precision(ref_list, retrieved_list))
        return np.array(mprecisions).mean()

    def cosine_distance(self, a, b):
        return 1 - np.dot(a, b) / (norm(a) * norm(b))

    def effective_err(self, queries: [str], ref: [[str]], retrieved: [[str]]):
        assert queries is not None
        assert self.feature_map is not None
        assert len(ref) == len(retrieved), \
            "reference and retrieved lists count not equal"
        assert len(ref)>0 and len(retrieved)>0, "empty ref/retr lists"
        errors = []
        for q, ref_list, retrieved_list in zip(queries, ref, retrieved):
            assert len(ref_list) == len(retrieved_list), \
                "ref list length is consistent with the retrieved list"
            vec_q = self.feature_map[q]
            vec_ref = self.feature_map[ref_list[0]]
            vec_retrieved = self.feature_map[retrieved_list[0]]
            
            numerator = self.cosine_distance(vec_q, vec_retrieved)
            denominator = self.cosine_distance(vec_q, vec_ref)
            assert numerator >= denominator, \
                f"{q}, ref:{ref_list[0]}, retr:{retrieved_list[0]}, {numerator}/{denominator}"
            if denominator == 0:
                err = numerator + 1.0
            else:
                err = numerator / denominator
            errors.append(err)
        return np.array(errors).mean()
