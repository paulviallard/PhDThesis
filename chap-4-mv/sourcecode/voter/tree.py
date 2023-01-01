import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
import torch

from voter.majority_vote import MajorityVote

###############################################################################


class Tree(BaseEstimator, ClassifierMixin):

    def __init__(self, dir, rand=None, max_depth=None):
        # Same tree as [1]
        # (see https://github.com/StephanLorenzen/MajorityVoteBounds/
        # blob/master/mvb/rfc.py)
        self.tree = DecisionTreeClassifier(
            criterion="gini",
            max_features="sqrt",
            min_samples_split=2,
            min_samples_leaf=1,
            max_depth=max_depth,
            random_state=rand)
        self.dir = dir

    def fit(self, X, y):
        """
        Run the algorithm CB-Boost

        Parameters
        ----------
        X: ndarray
            The inputs
        y: ndarray
            The labels
        """
        self.tree.fit(X, y)
        return self

    def output(self, X):
        """
        Get the output of the tree

        Parameters
        ----------
        X: tensor or ndarray
            The inputs
        """
        # X -> (size, nb_feature)
        assert ((isinstance(X, torch.Tensor) or isinstance(X, np.ndarray))
                and (len(X.shape) == 2))

        X_ = X
        if(isinstance(X, torch.Tensor)):
            X_ = X.detach().cpu().numpy()

        # We get the output
        out = self.dir*np.expand_dims(self.tree.predict(X_), 1)

        if(isinstance(X, torch.Tensor)):
            out = torch.tensor(out, device=X.device)
        return out


class TreeMV(MajorityVote):

    def __init__(self, x, y, nb_tree=100, max_depth=None, complemented=False):
        self.nb_tree = nb_tree
        self.max_depth = max_depth

        assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(x.shape) == 2 and len(y.shape) == 2 and
                x.shape[0] == y.shape[0] and
                y.shape[1] == 1 and x.shape[0] > 0)

        super().__init__(x, y)
        self.complemented = complemented
        self.fit()

    def fit(self):
        """
        Generate the forest (of trees)
        """
        super().fit()
        x, y = self.x_y_list

        # We generate the two directions
        # Note: the direction is used to generate the voter h and -h
        y_list = [+1]
        if(self.complemented):
            y_list = [+1, -1]

        # We generate the trees (for the first direction)
        y_ = y_list[0]
        for i in range(self.nb_tree):
            self.voter_list.append(Tree(y_, rand=i, max_depth=self.max_depth))
            self.voter_list[i].fit(x, y)

        # We generate the trees (for the second direction)
        if(self.complemented):
            y_ = y_list[1]
            for i in range(self.nb_tree):
                self.voter_list.append(
                    Tree(y_, rand=i, max_depth=self.max_depth))
                self.voter_list[self.nb_tree+i].fit(x, y)

        return self

###############################################################################

# References:
# [1] Second Order PAC-Bayesian Bounds for the Weighted Majority Vote
#     Andrés R. Masegosa, Stephan S. Lorenzen, Christian Igel, Yevgeny Seldin,
#     2020
