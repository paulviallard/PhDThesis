import torch
import cvxpy as cp
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import warnings
from voter.majority_vote import MajorityVote
from voter.majority_vote_diff import MajorityVoteDiff

###############################################################################


class MinCqLearner(BaseEstimator, ClassifierMixin):

    def __init__(self, majority_vote, mu):
        assert mu > 0 and mu <= 1
        self.mu = mu
        self.majority_vote = majority_vote
        self.mv = majority_vote

        self.device = "cpu"
        new_device = torch.device("cpu")
        if(torch.cuda.is_available() and self.device != "cpu"):
            new_device = torch.device(self.device)
        self.device = new_device

        if(isinstance(self.mv, MajorityVote)):
            self.mv_diff = MajorityVoteDiff(self.mv, self.device)
        self.mv_diff.to(self.device)

        # We assume that the majority vote is auto-complemented
        assert isinstance(self.mv, MajorityVote)

    def get_params(self, deep=True):
        return {"mu": self.mu, "majority_vote": self.mv}

    def fit(self, x, y):
        """
        Run the algorithm MinCq (Program 1 in [1])

        Parameters
        ----------
        x: ndarray
            The inputs
        y: ndarray
            The labels
        """
        # x -> (size, nb_feature)
        # y -> (size, 1)
        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(x.shape) == 2 and len(y.shape) == 2 and
                x.shape[0] == y.shape[0] and
                y.shape[1] == 1 and x.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        # We generate the prior and the posterior
        self.prior = ((1.0/len(self.mv.voter_list))
                      * np.ones((len(self.mv.voter_list), 1)))
        self.post = np.array(self.prior)

        out = self.mv.output(x)

        # We define Program 1
        nb_voter = len(self.post)
        nb_example = x.shape[0]

        M = (1.0/nb_example)*(out.T@out)
        m = np.mean(y*out, axis=0)
        a = (1/nb_voter)*np.sum(M, axis=0)

        post_ = cp.Variable(shape=(nb_voter, 1))

        prob = cp.Problem(
            cp.Minimize(cp.quad_form(
                post_,
                cp.Parameter(shape=M.shape, value=M, PSD=True))-a.T@post_),
            [post_ >= 0.0, post_ <= 1.0/(nb_voter),
             2.0*(m.T@post_)-np.mean(m) == self.mu
             ])

        # We solve Program 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                prob.solve()
            self.post = post_.value
        except cp.error.SolverError:
            self.post = None

        if(self.post is None):
            self.post = np.array(self.prior)
        else:
            self.post = 2.0*self.post-(1.0/nb_voter)
            self.__quasi_uniform_to_normal(out)

        self.mv_diff.post_.data = torch.log(torch.tensor(
            self.post)).float()
        self.mv_diff.prior.data = torch.tensor(self.prior).float()

    def __quasi_uniform_to_normal(self, out):
        """
        Convert a "quasi-uniform" posterior into a "normal" posterior
        """
        # We get a posterior distribution such that
        # (i) it preserves the margin and the disagreement
        # (ii) it minimizes the KL divergence
        post_ = cp.Variable(shape=self.post.shape)
        prob = cp.Problem(cp.Minimize(cp.sum(cp.kl_div(post_, self.prior))),
                          [out@post_ == out@self.post,
                           cp.sum(post_) == 1,
                           post_ >= 0])
        prob.solve()
        self.post = np.abs(post_.value)/np.sum(np.abs(post_.value))

    def __normal_to_quasi_uniform(self):
        """
        Apply Theorem 43 of [1] to convert a "normal" posterior
        into the "quasi-uniform" posterior
        """
        n = len(self.mv.voter_list)
        post_n = self.post[:n//2]
        post_2n = self.post[n//2:]
        post_n_ = (1/n) - (post_n-post_2n)/(n*np.max(np.abs(post_n-post_2n)))
        post_2n_ = (1/n) - (post_2n-post_n)/(n*np.max(np.abs(post_n-post_2n)))
        self.post = np.concatenate((post_n_, post_2n_), axis=0)
        self.quasi_uniform = True

    def predict(self, X):
        """
        Predict the label of the inputs
        X: ndarray
            The inputs
        """
        out = self.mv.output(X)
        pred = out@self.post
        pred = np.sign(pred)
        return pred


###############################################################################

# References:
# [1] From PAC-Bayes Bounds to Quadratic Programs for Majority Votes
#     François Laviolette, Mario Marchand, Jean-Francis Roy, 2011
