import time
import torch
from voter.majority_vote_diff import MajorityVoteDiff

###############################################################################


class MultipleMajorityVoteDiff(torch.nn.Module):

    def __init__(self, majority_vote_list, device):
        super().__init__()

        self.mv_list = majority_vote_list
        self.device = device

        self.post_ = []
        self.prior = []
        self.mv_diff_list = []
        for mv in self.mv_list:
            self.mv_diff_list.append(MajorityVoteDiff(mv, self.device))
            mv_diff = self.mv_diff_list[-1]
            self.post_.append(mv_diff.post_)
            self.prior.append(mv_diff.prior)
        self.post__ = torch.nn.ParameterList(self.post_)
        self.prior = torch.cat(self.prior)
        self.post_ = torch.cat(self.post_)
        self.y_unique = torch.tensor([])

    def forward(self, batch):
        """
        Forward the inputs in the majority vote

        Parameters
        ----------
        batch: dict
            The inputs
        """
        x_size = 0
        for i in range(1, len(self.mv_diff_list)+1):
            x_size += batch[f"x_{i}"].shape[0]

        self.margin = torch.ones((x_size, 1))
        self.margin = self.margin*(float('nan'))
        self.score = None

        y_unique_size = None
        self.kl = []
        i = 0
        x_size_tmp = 0
        while(i < len(self.mv_diff_list)):

            i_ = len(self.mv_diff_list)-i

            mv = self.mv_diff_list[i]
            batch_ = {
                "x": batch[f"x_{i_}"].view(batch[f"x_{i_}"].shape[0], -1),
                "y": batch[f"y_{i_}"]}
            mv(batch_)
            self.y_unique, _ = torch.sort(
                torch.unique(torch.concat([self.y_unique, mv.y_unique])))
            if(y_unique_size is None):
                y_unique_size = len(self.y_unique)

            if(len(self.y_unique) != y_unique_size
               or len(self.y_unique) != len(mv.y_unique)):
                for j in range(len(self.mv_diff_list)):
                    self.mv_diff_list[j].y_unique = self.y_unique.clone()
                y_unique_size = len(self.y_unique)
                self.score = None
                i = 0
                x_size_tmp = 0
            else:
                margin = mv.margin
                self.margin[x_size_tmp:x_size_tmp+len(batch_["x"])] = margin

                if(self.score is None):
                    self.score = mv.score
                else:
                    self.score = torch.concat(
                        (self.score, mv.score), axis=0)
                x_size_tmp += len(batch_["x"])
                i += 1

        pred = torch.max(self.score, axis=1, keepdims=True)[1]
        self.pred = self.y_unique[pred]
        self.KL()

    #  def predict(self, x=None):
    #
    #      if(not(is_forwarded)):
    #          for i in range(len(self.mv_diff_list)):
    #              mv = self.mv_diff_list[i]
    #              mv.predict(x)
    #
    #      for i in range(len(self.mv_diff_list)):
    #          mv = self.mv_diff_list[i]
    #          if(self.score is None):
    #              self.score = mv.score
    #              self.score = self.score.unsqueeze(0)
    #              self.score = self.score.repeat(
    #                  [len(self.mv_diff_list), 1, 1])
    #          else:
    #              self.score[i:i+1, :, :] = mv.score
    #
    #      self.score = torch.mean(self.score, axis=0)
    #      pred = torch.max(self.score, axis=1, keepdims=True)[1]
    #      self.pred = self.y_unique[pred]
    #      return self.pred

    def KL(self):
        self.kl = []
        for i in range(len(self.mv_diff_list)):
            mv = self.mv_diff_list[i]
            mv.KL()
            self.kl.append(mv.kl)
        self.kl = torch.tensor(self.kl)
        self.kl = torch.mean(self.kl)
