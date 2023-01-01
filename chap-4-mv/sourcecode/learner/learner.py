import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class Learner(BaseEstimator, ClassifierMixin):
    def __init__(self):
        super(Learner, self).__init__()

    def fit(self, X, y):
        raise NotImplementedError()

    def predict(self, X):
        pred = self.mv.predict(X)
        pred = np.expand_dims(pred, axis=1)
        return pred

    def predict_proba(self, X):
        return self.mv.predict_proba(X)
