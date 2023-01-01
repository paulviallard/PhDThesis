import math
import torch

from core.modules import Modules
from core.cocob_optim import COCOB
from learner.gradient_descent_learner import GradientDescentLearner
from voter.majority_vote_diff import MajorityVoteDiff
from voter.multiple_majority_vote_diff import MultipleMajorityVoteDiff
from voter.majority_vote import MajorityVote


###############################################################################


class CBoundMcAllesterLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch=1, m=1, batch_size=None, delta=0.05, t=100,
        writer=None
    ):
        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size,
                         writer=writer)
        self.m = m
        self.t = t
        self._optim = None
        self.delta = delta

        # Assume that we have the device
        if(isinstance(self.mv, MajorityVote)):
            self.mv_diff = MajorityVoteDiff(self.mv, self.device)
        elif(isinstance(self.mv, list)):
            self.mv_diff = MultipleMajorityVoteDiff(self.mv, self.device)
        else:
            raise RuntimeError(
                "self.mv must be MajorityVote or list")
        self.mv_diff.to(self.device)

        self.risk = Modules("Risk", self.mv_diff)
        self.disa = Modules("Disagreement", self.mv_diff)
        self.cbound = Modules(
            "CBoundMcAllester", self.mv_diff, m=self.m, delta=self.delta)
        self.zero_one_loss = Modules("ZeroOne", self.mv_diff).fit

    def __log_barrier(self, x):
        """
        Compute the log-barrier extension of [2]

        Parameters
        ----------
        x: tensor
            The constraint to optimize
        """
        assert isinstance(x, torch.Tensor) and len(x.shape) == 0
        # We use the
        if(x <= -1.0/(self.t**2.0)):
            return -(1.0/self.t)*torch.log(-x)
        else:
            return self.t*x - (1.0/self.t)*math.log(1/(self.t**2.0))+(1/self.t)

    def _optimize(self, batch):
        """
        Optimize the PAC-Bayesian bound of [3]

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

        # We optimize the PAC-Bayesian C-Bound of [3]
        cbound = self.cbound.fit(margin=margin)
        risk = self.risk.fit(margin=margin)
        disa = self.disa.fit(margin=margin)

        self._loss = cbound
        self._loss += self.__log_barrier(self.cbound.rD-0.5)

        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

        self._log["c-bound"] = cbound
        self._log["0-1 loss"] = self.zero_one_loss(y=y, pred=pred)

        if(self.writer is not None):
            self._write["risk_bound"] = self.cbound.rD.item()
            self._write["disa_bound"] = self.cbound.dD.item()
            self._write["c-bound"] = cbound
            self._write["loss"] = self._loss
            self._write["kl"] = kl.item()
            self._write["risk"] = risk.item()
            self._write["disa"] = disa.item()
            self._write["01_loss"] = self._log["0-1 loss"].item()

###############################################################################

# References:
# [1] Risk Bounds for the Majority Vote:
#     From a PAC-Bayesian Analysis to a Learning Algorithm
#     Pascal Germain, Alexandre Lacasse, Francois Laviolette,
#     Mario Marchand, Jean-Francis Roy, 2015
# [2] Constrained Deep Networks: Lagrangian Optimization
#     via Log-Barrier Extensions
#     Hoel Kervadec, Jose Dolz, Jing Yuan, Christian Desrosiers,
#     Eric Granger, Ismail Ben Ayed, 2019
# [3] A Column Generation Bound Minimization Approach with
#     PAC-Bayesian Generalization Guarantees
#     Jean-Francis Roy, Mario Marchand, François Laviolette, 2016
