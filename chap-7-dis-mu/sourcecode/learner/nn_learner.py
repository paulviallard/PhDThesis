import torch
import logging

from model.model import Model
from core.module import Module

from core.mh_optim import MH
from learner.sampling_learner import SamplingLearner

###############################################################################


class NNLearner():

    def __init__(
        self, model, model_kwargs, batch_size, epoch_size, epoch_mh,
        lr_sgd, lr_mh, alpha, device, writer=None,
    ):
        self.model_kwargs = model_kwargs
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.epoch_mh = epoch_mh
        self.lr_sgd = lr_sgd
        self.lr_mh = lr_mh
        self.writer = writer
        self.alpha = alpha

        self.device = torch.device('cpu')
        if(torch.cuda.is_available() and device != "cpu"):
            self.device = torch.device(device)

        self.model = Model(model, self.device, self.model_kwargs)
        self.model.to(self.device)

        # ------------------------------------------------------------------- #

        self.__loss = Module("BoundedCrossEntropy", self.model)
        self.__criteria = Module("ZeroOne", self.model)

        self.loss = self.__loss.fit
        self.criteria = self.__criteria.fit

        self.optim = MH(
            self.__loss.parameters(), lr=self.lr_sgd, alpha=self.alpha)
        self.learner = SamplingLearner(
            self.model, self.loss, self.optim,
            self.device, batch_size=self.batch_size,
            epoch_size=self.epoch_size, epoch_mh=self.epoch_mh,
            lr_sgd=self.lr_sgd, lr_mh=self.lr_mh,
            writer=self.writer, alpha=self.alpha)

        # ------------------------------------------------------------------- #

    def fit(self, X, y):
        self.__loss.m = len(X)
        logging.info("Learning ...\n")
        self.learner.fit(X, y)

    def save(self):
        return self.learner.save()

    def load(self, state_dict):
        return self.learner.load(state_dict)

    def output(self, X):
        return self.learner.output(X)

    def predict(self, X):
        return self.learner.predict(X)

    def get_measures(self, X, y):
        out = self.output(X)
        return self.learner.model.get_measures(out, y)

    def predict_proba(self, X):
        raise NotImplementedError

###############################################################################
