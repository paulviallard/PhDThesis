import math
import torch

from core.modules import Modules
from core.cocob_optim import COCOB
from learner.gradient_descent_learner import GradientDescentLearner
from voter.majority_vote_diff import MajorityVoteDiff
from voter.multiple_majority_vote_diff import MultipleMajorityVoteDiff
from voter.majority_vote import MajorityVote


###############################################################################


class CBoundJointLearner(GradientDescentLearner):

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

        self.joint = Modules("Joint", self.mv_diff)
        self.disa = Modules("Disagreement", self.mv_diff)
        self.cbound = Modules(
            "CBoundLacasse", self.mv_diff, m=self.m, delta=self.delta)
        self.zero_one_loss = Modules("ZeroOne", self.mv).fit

    def __log_barrier(self, x):
        """
        Compute the log-barrier extension of [2]

        Parameters
        ----------
        x: tensor
            The constraint to optimize
        """
        assert isinstance(x, torch.Tensor) and len(x.shape) == 0
        if(x <= -1.0/(self.t**2.0)):
            return -(1.0/self.t)*torch.log(-x)
        else:
            return self.t*x - (1.0/self.t)*math.log(1/(self.t**2.0))+(1/self.t)

    def __bound(self, kl, m, delta):
        """
        Compute the PAC-Bayesian bound of PAC-Bound 2 (see page 820 of [1])

        Parameters
        ----------
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        delta: float
            The confidence parameter of the bound
        """
        b = math.log((2.0*math.sqrt(m)+m)/delta)
        b = (1.0/m)*(2.0*kl+b)
        return b

    def __kl_tri(self, q1, q2, p1, p2):
        """
        Compute the KL divergence between two trinomials
        (see eq. (31) of [1])

        Parameters
        ----------
        q1: tensor
            The first parameter of the posterior trinomial distribution
        q2: tensor
            The second parameter of the posterior trinomial distribution
        p1: tensor
            The first parameter of the prior trinomial distribution
        p2: tensor
            The second parameter of the prior trinomial distribution
        """
        # For numerical stability
        q1 = torch.maximum(q1, torch.tensor(0.0, device=q1.device))
        q2 = torch.maximum(q2, torch.tensor(0.0, device=q2.device))
        p1 = torch.maximum(p1, torch.tensor(10**(-10), device=p1.device))
        p2 = torch.maximum(p2, torch.tensor(10**(-10), device=p2.device))

        kl = (torch.torch.special.xlogy(q1, q1)
              - torch.torch.special.xlogy(q1, p1))
        kl += (torch.torch.special.xlogy(q2, q2)
               - torch.torch.special.xlogy(q2, p2))
        kl += (torch.torch.special.xlogy((1-q1-q2), (1-q1-q2))
               - torch.torch.special.xlogy((1-q1-q2), (1-p1-p2)))

        return kl

    def __optimize_given_e_d(self, e, d, eS, dS, kl, m):
        """
        Optimize the posterior distribution given
        the joint error e and the disagreement d

        Parameters
        ----------
        e: float
            The (true) empirical error
        d: float
            The (true) disagreement
        eS: ndarray
            The empirical joint error
        dS: ndarray
            The empirical disagreement
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        e = torch.tensor(e, device=eS.device)
        d = torch.tensor(d, device=dS.device)

        b = self.__bound(kl, m, self.delta)
        self._loss = -self.__log_barrier(self.__kl_tri(dS, eS, d, e)-b)
        self._loss += self.__log_barrier((2.0*eS+dS)-1.0)

        # We compute the gradient descent step given (e,d)
        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

    def _optimize(self, batch):
        """
        Optimize the PAC-Bound 2 by gradient descent

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

        # We compute the C-Bound
        cbound = self.cbound.fit(margin=margin)
        joint = self.joint.fit(margin=margin)
        disa = self.disa.fit(margin=margin)

        # We optimize the outer minimization problem
        if(self.cbound.eD is not None and self.cbound.dD is not None):
            self.__optimize_given_e_d(
                self.cbound.eD, self.cbound.dD,
                self.cbound.eS, self.cbound.dS, kl, self.m)

        self._log["c-bound"] = cbound
        self._log["0-1 loss"] = self.zero_one_loss(y=y, pred=pred)

        if(self.writer is not None):
            self._write["joint_bound"] = self.cbound.eD
            self._write["disa_bound"] = self.cbound.dD
            self._write["c-bound"] = cbound
            self._write["loss"] = self._loss
            self._write["kl"] = kl.item()
            self._write["joint"] = joint.item()
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
# [3] Convex Optimization
#     Stephen Boyd, Lieven Vandenberghe, 2004
