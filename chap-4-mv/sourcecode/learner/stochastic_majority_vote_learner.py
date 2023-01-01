import torch

from core.cocob_optim import COCOB
from core.modules import Modules
from learner.gradient_descent_learner import GradientDescentLearner
from voter.sto_majority_vote_diff import StoMajorityVoteDiff
from voter.multiple_sto_majority_vote_diff import MultipleStoMajorityVoteDiff
from voter.majority_vote import MajorityVote

###############################################################################


class StochasticMajorityVoteLearner(GradientDescentLearner):

    def __init__(
        self, majority_vote, epoch, batch_size=None,
        m=1, delta=0.05, risk="exact", sigmoid_c=100, mc_draws=10, prior=1.0,
        writer=None
    ):

        super().__init__(majority_vote, epoch=epoch, batch_size=batch_size,
                         writer=writer)

        if(isinstance(self.mv, MajorityVote)):
            self.mv_diff = StoMajorityVoteDiff(
                self.mv, self.device, prior=prior, mc_draws=mc_draws,
                sigmoid_c=sigmoid_c, risk_name=risk)
        elif(isinstance(self.mv, list)):
            self.mv_diff = MultipleStoMajorityVoteDiff(
                self.mv, self.device, prior=prior, mc_draws=mc_draws,
                sigmoid_c=sigmoid_c, risk_name=risk)
        else:
            raise RuntimeError(
                "self.mv must be MajorityVote or list")

        self._optim = None

        self.m = m
        self.delta = delta

        self.risk = risk
        self.sigmoid_c = sigmoid_c
        self.prior = prior

        self.risk = Modules("Risk", self.mv_diff)
        self.bound = Modules(
            "BoundSto", self.mv_diff, m=self.m, delta=self.delta)
        self.zero_one_loss = Modules("ZeroOne", self.mv).fit

    def _optimize(self, batch):
        if(self._optim is None):
            self._optim = COCOB(self.mv_diff.parameters())

        if("x" in batch):
            batch["x"] = batch["x"].to(
                device=self.device, dtype=torch.float32)
            batch["y"] = batch["y"].to(
                device=self.device, dtype=torch.long)
            assert "y" in batch and isinstance(batch["y"], torch.Tensor)
            y = batch["y"]
        elif("x_1" in batch):
            i = 1
            y = None
            while(f"x_{i}" in batch):
                batch[f"x_{i}"] = batch[f"x_{i}"].to(
                    device=self.device, dtype=torch.float32)
                batch[f"y_{i}"] = batch[f"y_{i}"].to(
                    device=self.device, dtype=torch.long)
                assert f"y_{i}" in batch and isinstance(
                    batch[f"y_{i}"], torch.Tensor)
                if(y is None):
                    y = batch[f"y_{i}"]
                else:
                    y = torch.concat((batch[f"y_{i}"], y), axis=0)
                i += 1

        # We compute the prediction of the majority vote
        self.mv_diff(batch)
        pred = self.mv_diff.pred
        margin = self.mv_diff.margin
        kl = self.mv_diff.kl

        assert len(y.shape) == 2 and len(pred.shape) == 2
        assert (pred.shape[0] == y.shape[0] and pred.shape[1] == y.shape[1]
                and y.shape[1] == 1)
        assert len(kl.shape) == 0

        # We compute the bound
        bound = self.bound.fit(margin=margin)
        self._loss = bound
        risk = self.risk.fit(margin=margin)

        self._log["bound"] = self._loss
        self._log["0-1 loss"] = self.zero_one_loss(y=y, pred=pred)

        # We backward
        self._optim.zero_grad()
        self._loss.backward()
        self._optim.step()

        if(self.writer is not None):
            self._write["loss"] = self._loss.item()
            self._write["kl"] = kl.item()
            self._write["risk"] = risk.item()
            self._write["01_loss"] = self._log["0-1 loss"].item()

            if(isinstance(self.mv_diff, StoMajorityVoteDiff)):
                self._write["post_norm"] = torch.norm(
                    self.mv_diff.post_).item()
                self._write["post_grad_norm"] = torch.norm(
                    self.mv_diff.post_.grad).item()
            else:
                self._write["post_norm"] = {}
                self._write["post_grad_norm"] = {}
                for i in range(len(self.mv_diff.mv_list)):
                    self._write["post_norm"][i] = torch.norm(
                        self.mv_diff.mv_diff_list[i].post_).item()
                    self._write["post_grad_norm"][i] = torch.norm(
                        self.mv_diff.mv_diff_list[i].post_.grad).item()

###############################################################################
