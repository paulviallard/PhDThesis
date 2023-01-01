import torch
import numpy as np

###############################################################################


class MajorityVoteDiff(torch.nn.Module):

    def __init__(self, majority_vote, device):
        super().__init__()

        self.mv = majority_vote
        self.device = device

        self.prior = (1.0/len(self.mv.voter_list))*torch.ones(
            (len(self.mv.voter_list), 1), device=self.device)
        self.post_ = torch.nn.Parameter(
            self.prior.clone().to(self.device))
        self.y_unique = torch.tensor([])

    def forward(self, batch):
        """
        Forward the inputs in the majority vote

        Parameters
        ----------
        batch: dict
            The inputs
        """
        assert isinstance(self.prior, torch.Tensor)

        x = batch["x"]
        y = batch["y"]
        self.y_unique = torch.unique(torch.concat(
            [self.y_unique, torch.flatten(y)]))
        x = x.view(x.shape[0], -1)

        # We clamp to avoid inf values
        self.post_.data = torch.clamp(self.post_, -10**10, 10**10)

        # We get the posterior distribution (thanks to a softmax)
        self.post = torch.nn.functional.softmax(self.post_, dim=0)

        # We predict the examples with the voters
        self.out = self.mv.output(x)
        out_unique = torch.unique(self.out)

        self.y_unique, _ = torch.sort(torch.unique(
            torch.cat((self.y_unique, out_unique))))

        margin = (self.out == y).float()
        margin = 2.0*(margin-0.5)
        self.margin = (margin@self.post)

        self.predict(y=y, out=self.out)
        self.KL()

        # pred -> (size, 1)
        assert (len(self.pred.shape) == 2
                and self.pred.shape[0] == x.shape[0]
                and self.pred.shape[1] == 1)
        # out -> (size, nb_voter)
        assert (len(self.out.shape) == 2
                and self.out.shape[0] == x.shape[0]
                and self.out.shape[1] == self.post.shape[0])

        # margin -> (size, 1)
        assert (len(self.margin.shape) == 2
                and self.margin.shape[0] == x.shape[0]
                and self.margin.shape[1] == 1)

    def KL(self):

        self.kl = torch.special.xlogy(self.post, self.post).sum()
        self.kl = self.kl-torch.special.xlogy(self.post, self.prior).sum()

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
            score_ = (score_@self.post)

            if(self.score is None):
                self.score = score_
            else:
                self.score = torch.cat((self.score, score_), axis=1)

        pred = torch.max(self.score, axis=1, keepdims=True)[1]
        self.pred = self.y_unique[pred]
        return self.torch_to_numpy(ref, self.pred)
