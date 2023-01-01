import torch

from voter.majority_vote_diff import MajorityVoteDiff
from voter.multiple_majority_vote_diff import MultipleMajorityVoteDiff
from voter.majority_vote import MajorityVote
from sklearn.base import BaseEstimator, ClassifierMixin


###############################################################################


class NothingLearner(BaseEstimator, ClassifierMixin):

    def __init__(self, majority_vote):
        super().__init__()

        self.device = "cpu"
        new_device = torch.device("cpu")
        if(torch.cuda.is_available() and self.device != "cpu"):
            new_device = torch.device(self.device)
        self.device = new_device

        # Assume that we have the device
        if(isinstance(majority_vote, MajorityVote)):
            self.mv_diff = MajorityVoteDiff(majority_vote, self.device)
        elif(isinstance(majority_vote, list)):
            self.mv_diff = MultipleMajorityVoteDiff(majority_vote, self.device)
        else:
            raise RuntimeError(
                "self.mv must be MajorityVote or list")
        self.mv_diff.to(self.device)

    def fit(self, x, y):
        self.mv_diff.post_.data = torch.log(self.mv_diff.prior.clone())

###############################################################################
