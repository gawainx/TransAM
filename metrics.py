import numpy as np
from typing import Union
import torch
import pandas as pd


class KGLPMetrics:

    def __init__(self):
        self._hits_1 = 0.0
        self._hits_5 = 0.0
        self._hits_10 = 0.0
        self.count = 0.0
        self._mrr = 0.0
        self.ranks = list()

    @property
    def dataframe(self):
        return {
            'Hits@1':   self.hits_at_1,
            'Hits@5':   self.hits_at_5,
            'Hits@10':  self.hits_at_10,
            'MRR':      self.mrr,
            '#HITS@1':  self._hits_1,
            '#HITS@10': self._hits_10,
            '#HITS@5':  self._hits_5,
            '#MRR':     self._mrr,
            '#COUNT':   self.count
        }

    @property
    def hits_at_1(self):
        if self.count == 0:
            return 0.0
        return self._hits_1 / self.count

    @property
    def hits_at_5(self):
        if self.count == 0:
            return 0.0
        return self._hits_5 / self.count

    @property
    def hits_at_10(self):
        if self.count == 0:
            return 0.0
        return self._hits_10 / self.count

    @property
    def mrr(self):
        if self.count == 0:
            return 0.0
        return self._mrr / self.count

    def zeros(self):
        self._hits_1 = 0.0
        self._hits_5 = 0.0
        self._hits_10 = 0.0
        self.count = 0.0
        self._mrr = 0.0

    def clear(self):
        self.zeros()

    def copy(self):
        result = KGLPMetrics()
        result._hits_1 = self._hits_1
        result._hits_5 = self._hits_5
        result._hits_10 = self._hits_10
        result._mrr = self._mrr
        result.count = self.count
        result.ranks = self.ranks[:]
        return result

    @property
    def raw(self) -> str:
        result = dict()
        result['hits@1'] = self._hits_1
        result['hits@5'] = self._hits_5
        result['hits@10'] = self._hits_10
        result['count'] = self.count
        result['mrr'] = self._mrr
        result['ranks'] = self.ranks
        return str(result)

    def __str__(self):
        result = dict()
        result['Hits@1'] = f"{self.hits_at_1:.3f}"
        result['Hits@5'] = f"{self.hits_at_5:.3f}"
        result['Hits@10'] = f"{self.hits_at_10:.3f}"
        result['MRR'] = f"{self.mrr:.3f}"
        return str(result)

    def to_dict(self) -> dict:
        result = dict()
        result['Hits@1'] = f"{self.hits_at_1:.3f}"
        result['Hits@5'] = f"{self.hits_at_5:.3f}"
        result['Hits@10'] = f"{self.hits_at_10:.3f}"
        result['MRR'] = f"{self.mrr:.3f}"
        return result

    def __add__(self, other):
        if isinstance(other, KGLPMetrics):
            result = self.copy()
            result._hits_1 += other._hits_1
            result._hits_5 += other._hits_5
            result._hits_10 += other._hits_10
            result.count += other.count
            result._mrr += other._mrr
            result.ranks.extend(other.ranks)
            return result
        else:
            raise NotImplementedError

    def __gt__(self, other):
        if isinstance(other, KGLPMetrics):
            return self.mrr > other.mrr
        else:
            raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, KGLPMetrics):
            return self.mrr == other.mrr
        else:
            raise NotImplementedError

    def update(self, rank: int):
        self.ranks.append(rank)
        self.count += 1
        self._mrr += 1.0 / rank
        if rank <= 1:
            self._hits_1 += 1
            self._hits_10 += 1
            self._hits_5 += 1
        elif rank <= 5:
            self._hits_5 += 1
            self._hits_10 += 1
        elif rank <= 10:
            self._hits_10 += 1
        else:
            return

    def perform_metrics_numpy(self, scores: np.ndarray,
                              golden_idx: int = 0, find_max: bool = True):

        sort = list(np.argsort(scores, kind='stable'))  # [::-1]  # reverse a list
        if find_max:
            sort = sort[::-1]
        rank = sort.index(golden_idx) + 1

        self.update(rank)

    def perform_metrics(self, scores: Union[np.ndarray, torch.Tensor, list],
                        golden_idx: int = 0, find_max: bool = True):
        if isinstance(scores, np.ndarray):
            self.perform_metrics_numpy(scores, golden_idx, find_max)
        elif isinstance(scores, torch.Tensor):
            s = scores.detach().cpu().numpy()
            self.perform_metrics_numpy(s, golden_idx, find_max)
        else:
            s = np.array(scores)
            self.perform_metrics_numpy(s, golden_idx, find_max)


class EvalLogItem:
    def __init__(self, metrics: KGLPMetrics, epoch: int, ckpt_fn: str):
        self.metrics = metrics
        self.epoch = epoch
        self.ckpt = ckpt_fn

    def __eq__(self, other):
        if isinstance(other, EvalLogItem):
            return other.metrics == self.metrics and self.epoch == other.epoch
        else:
            raise NotImplementedError

    def __ge__(self, other):
        if isinstance(other, EvalLogItem):
            return self.metrics >= other.metrics

    def __lt__(self, other):
        if isinstance(other, EvalLogItem):
            return self.metrics < other.metrics
        else:
            raise NotImplementedError
