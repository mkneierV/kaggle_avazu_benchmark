import logging
import random

import cytoolz.itertoolz as toolz
import numpy as np
import scipy as sp

from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, TransformerMixin


class PartialFitter(BaseEstimator, TransformerMixin):
    def __init__(self,
                 steps,
                 batch_size=2000,
                 n_iter=10,
                 logging=False,
                 neg_rate=None):

        self.batch_size = batch_size
        self.steps = steps
        self.logging = logging
        self.n_iter = n_iter
        self.neg_rate = neg_rate

    def partial_fit(self, X, y):

        for i, batch in enumerate(toolz.partition_all(self.batch_size, X)):
            print 'batch {}'.format(i)

            rows = []
            response = []

            for row in batch:

                try:
                    row_y = y.next()
                    if self.check_response(row_y):
                        row = self._transform(row)
                        rows.append(row)
                        response.append(row_y)
                except Exception as e:
                    if self.logging:
                        logging.exception(e)
                    pass

            shuffledRange = range(len(rows))
            # need to shuffle data during each iteration
            for _ in range(self.n_iter):
                random.shuffle(shuffledRange)
                batch_data = sp.sparse.vstack([rows[i] for i in shuffledRange])
                shuffled_response = [response[i] for i in shuffledRange]
                self.steps[-1].partial_fit(batch_data, shuffled_response, classes=[0, 1])

    def predict_proba(self, newX):

        all_preds = []
        for batch in toolz.partition_all(self.batch_size, newX):
            pred_rows = []
            for newrow in batch:
                newrow = self._transform(newrow)
                pred_rows.append(newrow)
            test_data = sp.sparse.vstack(pred_rows)
            all_preds.append(self.steps[-1].predict_proba(test_data))

        return np.vstack(all_preds)

    def _transform(self, datapoint):
        for t in self.steps[:-1]:
            # Requiers stateless transformers
            datapoint = t.transform([datapoint])
            return datapoint

    def check_response(self, response_i):
        if self.neg_rate is not None:
            return (response_i == 1 or np.random.random() < self.neg_rate)
        else:
            return True


def llfun(act, pred):
    p_true = pred[:, 1]
    epsilon = 1e-15
    p_true = sp.maximum(epsilon, p_true)
    p_true = sp.minimum(1 - epsilon, p_true)
    ll = sum(act * sp.log(p_true) + sp.subtract(1, act) * sp.log(sp.subtract(1, p_true)))
    ll = ll * -1.0 / len(act)
    return ll


def _make_scorer(score_function):
    return make_scorer(llfun,
                       greater_is_better=False,
                       needs_proba=True)


ll_scorer = _make_scorer(llfun)
