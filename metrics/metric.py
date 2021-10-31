import numpy as np
import pandas as pd
import sklearn.metrics
import torch

"""Metric tracker for training and some metrics for testing.
"""

class MetricTracker:
    def __init__(self, *keys, writer=None, mode='/'):

        self.writer = writer
        self.mode = mode + '/'
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1, writer_step=1):
        if self.writer is not None:
            self.writer.add_scalar(self.mode + key, value, writer_step)
        self._data.total[key] += value
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def update_all_metrics(self, values_dict, n=1, writer_step=1):
        for key in values_dict:
            self.update(key, values_dict[key], n, writer_step)

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def calc_all_metrics(self):
        """Calculates string with all the metrics
        """
        s = ''
        d = dict(self._data.average)
        for key in dict(self._data.average):
            s += f'{key} {d[key]:7.4f}\t'
        return s

    def print_all_metrics(self):
        s = ''
        d = dict(self._data.average)
        for key in dict(self._data.average):
            s += "{} {:.4f}\t".format(key, d[key])
        return s



def confusion_matrix(predictions, targets):
    return sklearn.metrics.confusion_matrix(targets, predictions)


def sensitivity(cnf_matrix):
    """ie. recall
    """
    # print(cnf_matrix)
    eps = 1e-7
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    return TP / (TP + FN + eps)


def positive_predictive_value(cnf_matrix):
    """ie. precision
    """
    # print(cnf_matrix)
    eps = 1e-7
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
 
    return TP / (TP + FP + eps)
