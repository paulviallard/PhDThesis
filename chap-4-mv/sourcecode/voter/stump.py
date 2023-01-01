import numpy as np
import torch
from sklearn.base import BaseEstimator, ClassifierMixin

from voter.majority_vote import MajorityVote

###############################################################################


class DecisionStump(BaseEstimator, ClassifierMixin):

    def __init__(self, feature, threshold, dir):

        self.feature = feature
        self.threshold = threshold
        # Note: the direction is used to generate the voter h and -h
        self.dir = dir

        self.fit()

    def fit(self):
        return self

    def output(self, x):
        """
        Get the output of the tree

        Parameters
        ----------
        x: tensor or ndarray
            The inputs
        """
        # x -> (size, nb_feature)
        assert ((isinstance(x, torch.Tensor) or isinstance(x, np.ndarray))
                and (len(x.shape) == 2))

        # We get the output
        if(isinstance(x, torch.Tensor)):
            out = x[:, self.feature].unsqueeze(1)
            out = self.dir*(2.0*(out > self.threshold).float()-1.0)
        else:
            out = np.expand_dims(x[:, self.feature], 1)
            out = self.dir*(2.0*(out > self.threshold).astype(float)-1.0)
        return out


class DecisionStumpMV(MajorityVote):

    def __init__(
        self, x, y,
        nb_per_attribute=10, complemented=True
    ):
        self.nb_per_attribute = nb_per_attribute

        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(x.shape) == 2 and len(y.shape) == 2 and
                x.shape[0] == y.shape[0] and
                y.shape[1] == 1 and x.shape[0] > 0)
        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        super().__init__(
            x, y, complemented=complemented)

        self.fit()

    def fit(self):
        """
        Generate the forest (of stumps)
        """
        # x -> (size, nb_feature)
        # y -> (size, 1)
        super().fit()
        x, y = self.x_y_list
        x_min_list = np.min(x, axis=0)
        x_max_list = np.max(x, axis=0)
        gap_list = (x_max_list-x_min_list)/(self.nb_per_attribute + 1)

        # We generate the two directions
        # Note: the direction is used to generate the voter h and -h
        dir_list = [+1]
        if(self.complemented):
            dir_list = [+1, -1]

        # For each direction
        for dir in dir_list:
            # For each feature
            for i in range(len(x[0])):
                gap = gap_list[i]
                x_min = x_min_list[i]

                if gap == 0:
                    continue
                # We generate "self.nb_per_attribute" decision stumps
                for t in range(self.nb_per_attribute):
                    self.voter_list.append(
                        DecisionStump(i, x_min+gap*(t+1), dir))

        return self
