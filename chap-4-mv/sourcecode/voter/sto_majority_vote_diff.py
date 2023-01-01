import torch
import numpy as np
from core.beta_inc import BetaInc
from torch.distributions.dirichlet import Dirichlet as Dir


class StoMajorityVoteDiff(torch.nn.Module):

    def __init__(self, majority_vote, device, mc_draws=10, prior=1.0,
                 sigmoid_c=100, risk_name="exact"):

        super(StoMajorityVoteDiff, self).__init__()

        self.num_voters = len(majority_vote.voter_list)

        self.mv = majority_vote
        self.device = device

        # prior
        self.prior = torch.ones(self.num_voters)*prior

        #  self.voters = voters
        self.mc_draws = mc_draws
        self.sigmoid_c = sigmoid_c
        # uniform draws in (0, 2]
        #  post = torch.rand(self.num_voters) * 2 + 1e-9
        post = self.prior.detach().clone()
        # use log (and apply exp(post) later so that
        # posterior parameters are always positive)
        self.post_ = torch.nn.Parameter(torch.log(post), requires_grad=True)

        if(risk_name == "exact"):
            self.__margin_fun = self.__margin
        elif(risk_name == "MC"):
            self.__margin_fun = self.__approximated_margin
        else:
            raise RuntimeError("risk_name must be exact or MC")

        self.y_unique = torch.tensor([])

    def forward(self, batch):

        x = batch["x"]
        x = x.view(x.shape[0], -1)
        y = batch["y"]
        self.out = self.mv.output(x)

        self.KL()
        self.margin = self.__margin_fun(self.out, y)
        self.predict(y=y, out=self.out)

    def __margin(self, out, y):
        self.post = torch.exp(self.post_)

        correct = torch.where(
            y == out, self.post, torch.zeros(1)).sum(1)
        wrong = torch.where(
            y != out, self.post, torch.zeros(1)).sum(1)

        m = torch.cat([BetaInc.apply(
            c, w, torch.tensor(0.5)).unsqueeze(0) for c, w in zip(
            correct, wrong)])
        m = 1.0-2.0*torch.unsqueeze(m, 1)
        return m

    def __approximated_margin(self, out, y):
        mv_post_list = Dir(torch.exp(self.post_)).rsample((self.mc_draws,))

        m = torch.stack([torch.where(y != out, post, torch.zeros(1)).sum(1)
                         for post in mv_post_list])
        m = torch.mean(torch.sigmoid(self.sigmoid_c*(m-0.5)), axis=0)
        m = 1.0-2.0*torch.unsqueeze(m, 1)
        return m

    def KL(self):
        # Kullback-Leibler divergence between two Dirichlets
        self.post = torch.exp(self.post_)
        self.kl = torch.lgamma(self.post.sum())-torch.lgamma(self.post).sum()
        self.kl -= torch.lgamma(self.prior.sum())-torch.lgamma(
            self.prior).sum()
        self.kl += torch.sum((self.post - self.prior) * (
            torch.digamma(self.post)-torch.digamma(self.post.sum())))

    def numpy_to_torch(self, *var_list):
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(var_list[i], np.ndarray)):
                new_var_list.append(torch.tensor(var_list[i]))
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

    def torch_to_numpy(self, ref, *var_list):
        # Note: elements in var_list are considered as tensor
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(ref, np.ndarray)
               and isinstance(var_list[i], torch.Tensor)):
                new_var_list.append(var_list[i].detach().numpy())
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

    def predict(self, x=None, y=None, out=None):

        ref = x
        x, y, out = self.numpy_to_torch(x, y, out)

        if(out is None):
            self.out = self.mv.output(x)
        mv_post = Dir(torch.exp(self.post_)).rsample().unsqueeze(1)

        if(y is not None):
            self.y_unique, _ = torch.sort(torch.unique(
                torch.concat([
                    torch.flatten(self.out), torch.flatten(y), self.y_unique]))
            )
        else:
            self.y_unique, _ = torch.sort(torch.unique(
                torch.concat([
                    torch.flatten(self.out), self.y_unique]))
            )

        self.score = None
        for y_ in self.y_unique:
            score_ = (self.out == y_).float()
            score_ = (score_@mv_post)

            if(self.score is None):
                self.score = score_
            else:
                self.score = torch.cat((self.score, score_), axis=1)

        pred = torch.max(self.score, axis=1, keepdims=True)[1]
        self.pred = self.y_unique[pred]
        return self.torch_to_numpy(ref, self.pred)
