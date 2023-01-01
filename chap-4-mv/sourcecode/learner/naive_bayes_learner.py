import torch
import numpy as np
import warnings
from voter.majority_vote_diff import MajorityVoteDiff
from voter.majority_vote import MajorityVote


class NaiveBayesLearner():
    """
    Empirical frequentist or bayesian Naive Bayes classifier
    """
    def __init__(self, majority_vote, frequentist=False):

        self.device = "cpu"
        new_device = torch.device("cpu")
        if(torch.cuda.is_available() and self.device != "cpu"):
            new_device = torch.device(self.device)
        self.device = new_device

        self.frequentist = frequentist
        if(isinstance(majority_vote, MajorityVote)):
            self.mv_diff = MajorityVoteDiff(majority_vote, self.device)
        else:
            raise RuntimeError(
                "majority_vote must be MajorityVote")

    def fit(self, x, y):

        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(x.shape) == 2 and len(y.shape) == 2 and
                x.shape[0] == y.shape[0] and
                y.shape[1] == 1 and x.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        self.prior = ((1.0/len(self.mv_diff.mv.voter_list))
                      * np.ones((len(self.mv_diff.mv.voter_list), 1)))

        n = len(x)
        batch = {"x": torch.tensor(x), "y": torch.tensor(y)}
        x, y = batch["x"], batch["y"]

        batch["x"] = batch["x"].to(
            device=self.device, dtype=torch.float32)
        batch["y"] = batch["y"].to(
            device=self.device, dtype=torch.long)

        self.mv_diff(batch)
        out = self.mv_diff.out

        num_corrects = torch.where(
            y == out, torch.tensor(1), torch.tensor(0))
        num_corrects = torch.sum(num_corrects, axis=0).unsqueeze(1)

        if self.frequentist:
            p = num_corrects / n
            self.post = torch.log(p / (1 - p))
        else:
            # bayesian, with alpha = 1, beta = 1
            self.post = torch.log(
                (1 + num_corrects)/(1 + n - num_corrects))

        n = len(self.post)
        self.post[self.post >= 0] = 2.0*self.post[self.post >= 0]
        self.post[self.post < 0] = 0.0
        self.post = self.post/torch.sum(self.post)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.post = np.log(self.post)
        self.post[np.isinf(self.post)] = -10**10

        #  print(self.post)
        #  print(self.post[:n//2]+self.post[n//2:])
        #  self.post = self.post/np.max(np.abs(self.post))
        #  print(self.post)
        #  self.post = np.log(self.post)
        #  print(self.post)
        self.mv_diff.post_.data = torch.tensor(self.post).float()
        self.mv_diff.prior.data = torch.tensor(self.prior).float()

###############################################################################

# References:
# [1] A Finite Sample Analysis of the Naive Bayes Classifier
#     Daniel Berend and Aryeh Kontorovich,
#     2015
