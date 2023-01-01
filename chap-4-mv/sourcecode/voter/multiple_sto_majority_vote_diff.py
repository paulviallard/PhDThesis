import torch
from voter.sto_majority_vote_diff import StoMajorityVoteDiff
from voter.multiple_majority_vote_diff import MultipleMajorityVoteDiff

###############################################################################


class MultipleStoMajorityVoteDiff(MultipleMajorityVoteDiff):

    def __init__(self, majority_vote_list, device,
                 sigmoid_c=100, mc_draws=10, prior=1.0, risk_name="exact"):

        super(MultipleStoMajorityVoteDiff.__bases__[0], self).__init__()

        self.mv_list = majority_vote_list
        self.device = device
        self.mc_draws = mc_draws
        self.sigmoid_c = sigmoid_c
        self.prior = prior

        self.post_ = []
        self.prior = []
        self.mv_diff_list = []
        for mv in self.mv_list:
            self.mv_diff_list.append(StoMajorityVoteDiff(
                mv, self.device, mc_draws=mc_draws, sigmoid_c=sigmoid_c,
                prior=prior, risk_name=risk_name))
            mv_diff = self.mv_diff_list[-1]
            self.post_.append(mv_diff.post_)
            self.prior.append(mv_diff.prior)
        self.post__ = torch.nn.ParameterList(self.post_)
        self.post_ = torch.cat(self.post_)

        self.y_unique = torch.tensor([])
