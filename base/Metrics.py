"""
@author : Semsi Yigit Ozgumus
"""

import numpy as np

class Metrics(object):
    """
    Class for the evaluation metrics
    """
    def __init__(self):
        pass

    def precision(self,is_relevant):
        precision_score = np.sum(is_relevant, dtype=np.float32) / len(is_relevant)
        assert 0 <= precision_score <= 1, precision_score
        return precision_score

    def recall(self,is_relevant,position_items):
        recall_score = np.sum(is_relevant, dtype=np.float32)/ position_items.shape[0]
        assert 0 <= recall_score <= 1, recall_score
        return recall_score

    def map(self,is_relevant,position_items):
        p_at_k = is_relevant * np.cumsum(is_relevant,dtype=np.float32)\
                 / (1 + np.arange(is_relevant.shape[0]))
        map_score = np.sum(p_at_k) / np.min([position_items.shape[0], is_relevant.shape[0]])
        assert 0 <= map_score <= 1, map_score
        return map_score