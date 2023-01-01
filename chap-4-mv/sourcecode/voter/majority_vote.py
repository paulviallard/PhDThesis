import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch

###############################################################################


class MajorityVote(BaseEstimator, ClassifierMixin):

    def __init__(self, x, y, complemented=False, quasi_uniform=False):
        assert (isinstance(complemented, bool)
                and isinstance(quasi_uniform, bool))

        self.x = x
        self.y = y

        self.complemented = complemented
        self.prior = None
        self.post = None
        self.x_y_list = (np.array(x), np.array(y))
        self.fitted = False

    def fit(self):
        """
        Generate the voters (the function is completed in stump.py and tree.py)
        """
        self.fitted = True
        self.voter_list = []

        return self

    def output(self, X):
        """
        Get the output of each voter for the inputs

        Parameters
        ----------
        X: ndarray
            The inputs
        """
        out = None

        for voter in self.voter_list:

            if(out is None):
                out = voter.output(X)
            elif(isinstance(X, torch.Tensor)):
                out = torch.cat((out, voter.output(X)), dim=1)
            else:
                out = np.concatenate((out, voter.output(X)), 1)
        return out


###############################################################################

# References:
# [1] Risk Bounds for the Majority Vote:
#     From a PAC-Bayesian Analysis to a Learning Algorithm
#     Pascal Germain, Alexandre Lacasse, Francois Laviolette,
#     Mario Marchand, Jean-Francis Roy, 2015
