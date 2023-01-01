import torch

from core.modules import Modules
from core.cocob_optim import COCOB
from learner.gradient_descent_learner import GradientDescentLearner
from voter.majority_vote_diff import MajorityVoteDiff
from voter.multiple_majority_vote_diff import MultipleMajorityVoteDiff
from voter.majority_vote import MajorityVote


###############################################################################


class BoundRiskLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, m=1, batch_size=None, delta=0.05, t=100,
        writer=None
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size,
                         writer=writer)
        self.t = t
        self._optim = None
        self.delta = delta
        self.m = m

        # Assume that we have the device
        if(isinstance(self.mv, MajorityVote)):
            self.mv_diff = MajorityVoteDiff(self.mv, self.device)
        elif(isinstance(self.mv, list)):
            self.mv_diff = MultipleMajorityVoteDiff(self.mv, self.device)
        else:
            raise RuntimeError(
                "self.mv must be MajorityVote or list")
        self.mv_diff.to(self.device)

        self.bound = Modules(
            "BoundRisk", self.mv_diff, m=self.m, delta=self.delta)
        self.risk = Modules("Risk", self.mv_diff)
        self.zero_one_loss = Modules("ZeroOne", self.mv).fit

    def _optimize(self, batch):
        """
        Optimize the PAC-Bound 0 by gradient descent (see [1])

        Parameters
        ----------
        batch: dict
            The examples of the batch
        """
        if(self._optim is None):
            self._optim = COCOB(self.mv_diff.parameters())

        batch["x"] = batch["x"].to(
            device=self.device, dtype=torch.float32)
        batch["y"] = batch["y"].to(
            device=self.device, dtype=torch.long)

        # We compute the prediction of the majority vote
        self.mv_diff(batch)
        pred = self.mv_diff.pred
        margin = self.mv_diff.margin
        kl = self.mv_diff.kl

        assert "y" in batch and isinstance(batch["y"], torch.Tensor)
        y = batch["y"]

        assert len(y.shape) == 2 and len(pred.shape) == 2
        assert (pred.shape[0] == y.shape[0] and pred.shape[1] == y.shape[1]
                and y.shape[1] == 1)
        assert len(kl.shape) == 0

        # We compute the bound
        risk = self.risk.fit(margin=margin)
        bound = self.bound.fit(margin=margin)
        self._loss = bound

        # We backward
        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

        self._log["bound"] = self._loss
        self._log["0-1 loss"] = self.zero_one_loss(y=y, pred=pred)

        if(self.writer is not None):
            self._write["bound"] = self._loss.item()
            self._write["kl"] = kl.item()
            self._write["risk"] = risk.item()
            self._write["01_loss"] = self._log["0-1 loss"].item()

###############################################################################

# References:
# [1] Risk bounds for the majority vote:
#     from a PAC-Bayesian analysis to a learning algorithm
#     Pascal Germain, Alexandre Lacasse, François Laviolette,
#     Mario Marchand, Jean-Francis Roy, 2015
