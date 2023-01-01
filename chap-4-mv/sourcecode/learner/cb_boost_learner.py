import torch
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

from voter.majority_vote import MajorityVote
from voter.majority_vote_diff import MajorityVoteDiff

###############################################################################


class CBBoostLearner(BaseEstimator, ClassifierMixin):
    def __init__(
        self, majority_vote, max_iter=200, twice_same=True
    ):
        super(CBBoostLearner, self).__init__()

        self.mv = majority_vote
        assert isinstance(self.mv, MajorityVote)

        self.device = "cpu"
        new_device = torch.device("cpu")
        if(torch.cuda.is_available() and self.device != "cpu"):
            new_device = torch.device(self.device)
        self.device = new_device

        if(isinstance(self.mv, MajorityVote)):
            self.mv_diff = MajorityVoteDiff(self.mv, self.device)
        self.mv_diff.to(self.device)

        self.max_iter = max_iter
        self.twice_same = twice_same

    def fit(self, x, y):
        """
        Run the algorithm CB-Boost (Algorithm 1 in [1])

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

        # We compute the output matrix
        out = self.mv.output(x)

        # We initialize the list of selected voters
        self.selected_voter_list = []

        # We compute the individual margin
        y_out_matrix = y*out

        # Get n (the number of voters)
        n = y_out_matrix.shape[1]

        # We initialize the posterior
        # We generate the prior and the posterior
        self.prior = ((1.0/len(self.mv.voter_list))
                      * np.ones((len(self.mv.voter_list), 1)))
        self.post = np.zeros(self.prior.shape)

        # Initialize the majority vote (Lines 3 and 4)
        self.get_first_voter(y_out_matrix)

        # For each iteration (Line 5)
        it = n-1
        if(self.max_iter is not None and self.max_iter < n-1):
            it = self.max_iter - 1
        for k in range(it):

            # We get a new voter with its weight (and stop otherwise)
            # (From Line 6 to 13)
            if(not(self.get_new_voter(y, out, y_out_matrix))):
                break

        # We normalize (and update) the weights
        self.post = self.post/np.sum(self.post)
        self.mv.post = self.post

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.post = np.log(self.post)
        self.post[np.isinf(self.post)] = -10**10
        self.mv_diff.post_.data = torch.tensor(self.post).float()
        self.mv_diff.prior.data = torch.tensor(self.prior).float()

    def get_first_voter(self, y_out_matrix):
        """
        We select the first voter of the majority vote
        (Lines 3 and 4 of Algorithm 1 in [1])

        Parameters
        ----------
        y_out_matrix: tensor
            The matrix of the individual margins
        """
        # We compute the margin for each voter
        margin_matrix = np.sum(y_out_matrix, axis=0)

        # We get the voter with the highest margin (and set its weight to 1)
        margin_argmax = np.argmax(margin_matrix)
        self.selected_voter_list.append(margin_argmax)
        self.post[margin_argmax] = 1.0

    def get_new_voter(self, y, out, y_out_matrix):
        """
        We get a new voter using Theorem 3 in [1] (From Line 6 to 13)

        Parameters
        ----------
        y: tensor
            The label of the examples
        out: tensor
            The outputs of the majority vote
        y_out_matrix: tensor
            The matrix of the individual margins

        Returns
        -------
        bool
            True if a voter has been added and False otherwise
        """
        m = y_out_matrix.shape[0]

        # We compute F_k
        F_k = out@self.post

        # We compute Definitions 8, 9 and 10 for h_k and F_k
        tau_F_k_h_k = np.mean(F_k*out, axis=0)
        gamma_F_k = np.mean(y*F_k, axis=0)
        gamma_h_k = np.mean(y_out_matrix, axis=0)
        mu_F_k = np.mean(F_k**2.0, axis=0)

        # We generate the test for the optimal alpha_k (Line 8 of Algorithm 1)
        test_alpha = (tau_F_k_h_k-gamma_F_k/gamma_h_k)

        # We compute the alpha_k optimal when test_alpha < 0 (Line 9 of Algo 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            alpha_k = ((gamma_h_k*mu_F_k - gamma_F_k*tau_F_k_h_k)
                       / (gamma_F_k-gamma_h_k*tau_F_k_h_k))

        # We intialize the C-Bounds with the different alpha_k
        # (that respect the conditions)
        c_bound_k = np.zeros(y_out_matrix.shape[1])
        if not self.twice_same:
            c_bound_k[self.selected_voter_list] = np.inf
        c_bound_k[gamma_h_k <= 0] = np.inf
        c_bound_k[test_alpha >= 0] = np.inf
        c_bound_k[alpha_k < 0] = np.inf

        alpha_k = np.expand_dims(alpha_k, 1)

        c_bound_k_ = y*(F_k+(out*alpha_k.T))
        c_bound_k_ = ((np.sum(c_bound_k_, axis=0)**2.0)
                      / (np.sum(c_bound_k_**2.0, axis=0)))
        c_bound_k_ = 1.0-(1.0/m)*c_bound_k_
        c_bound_k[c_bound_k != np.inf] = c_bound_k_[c_bound_k != np.inf]

        # We compute the argmin of Line 12 in Algo 1
        alpha_argmin = np.argmin(c_bound_k)

        # If all C-bound are inf => there is no possible improvement, we stop
        if(c_bound_k[alpha_argmin] == np.inf):
            return False

        # We update the weights of the classifer F_(k+1)
        self.post[alpha_argmin] = self.post[alpha_argmin]+alpha_k[alpha_argmin]
        self.selected_voter_list.append(alpha_argmin)

        return True


###############################################################################

# We thank Baptise Bauvin for sharing his version of CB-Boost
# during the development of this source code.

# References:
# [1] Fast greedy C-bound minimization with guarantees,
#     Baptiste Bauvin, Cécile Capponi, Jean-Francis Roy, François Laviolette,
#     2020
