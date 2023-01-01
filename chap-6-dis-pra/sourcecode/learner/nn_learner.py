import torch
import numpy as np
import logging

from model.model import Model
from core.modules import Modules

from core.numpy_dataset import NumpyDataset
from learner.optimize_gradient_descent_learner import (
    OptimizeGradientDescentLearner)
from learner.early_stopping_learner import EarlyStoppingLearner

from sklearn.metrics import zero_one_loss


###############################################################################

class NNLearner():

    def __init__(
        self, model, var, batch_size, prior, device,
        step=None, load=None,
        prior_lr=None, post_lr=None, prior_epoch=None, post_epoch=None,
        delta=None, bound=None, writer=None,
    ):
        self.var = var
        self.batch_size = batch_size

        self.prior = float(prior)

        self.device = torch.device('cpu')
        if(torch.cuda.is_available() and device != "cpu"):
            self.device = torch.device(device)

        self.model = Model(model, self.device, var=self.var)
        self.model.to(self.device)

        self.criteria = zero_one_loss

        self.step = step
        self._load = load

        self.writer = writer

        # PRIOR ------------------------------------------------------------- #

        if(step is None or step == "prior"):
            self.prior_epoch = prior_epoch
            self.prior_lr = prior_lr

            self.prior_loss = Modules("BoundedCrossEntropyLoss", model).fit
            self.prior_optim = torch.optim.Adam(
                self.model.parameters(), lr=self.prior_lr)

            self.prior_learner = OptimizeGradientDescentLearner(
                self.model, self.prior_loss, self.criteria, self.prior_optim,
                self.device, epoch=1, batch_size=self.batch_size)
            self.prior_early_learner = EarlyStoppingLearner(
                self.prior_learner, self.criteria, val_epoch=self.prior_epoch)

        # POSTERIOR --------------------------------------------------------- #

        if(step is None or step == "post"):

            self.prior_epoch = prior_epoch
            self.post_epoch = post_epoch
            self.post_lr = post_lr

            self.delta = delta
            self.bound = bound

            self.__post_loss = Modules("BoundedCrossEntropyLoss", model).fit
            # NOTE: the number of example in the bound
            # will be modified during fit
            if(self.bound == "ours"
               or self.bound == "rivasplata"
               or self.bound == "catoni"
               or self.bound == "blanchard"
               ):
                self.__post_loss = Modules(
                    "Bound", self.model,
                    self.__post_loss, 1, self.delta,
                    T=self.prior_epoch, bound=self.bound)
            else:
                raise RuntimeError(
                    "bound is either ours; rivasplata; catoni or blanchard")

            self.post_loss = self.__post_loss.fit
            self.post_optim = torch.optim.Adam(
                self.__post_loss.parameters(), lr=self.post_lr)

            self.post_learner = OptimizeGradientDescentLearner(
                self.model, self.post_loss, self.criteria, self.post_optim,
                self.device, epoch=self.post_epoch, batch_size=self.batch_size)

        # ------------------------------------------------------------------- #

    def fit(self, X, y):

        size_prior = int(self.prior*len(X))
        x_prior = X[:size_prior, :]
        x_post = X[size_prior:, :]
        y_prior = y[:size_prior]
        y_post = y[size_prior:]

        if((self.step is None or self.step == "prior") and size_prior == 0):
            if(self.writer is not None):
                self.writer.write("state_dict", self.save())
            return
        else:
            if(size_prior != 0):
                x_prior = np.reshape(x_prior, (x_prior.shape[0], -1))
        x_post = np.reshape(x_post, (x_post.shape[0], -1))

        if(self.step is None or self.step == "prior"):

            logging.info("Running prior learning ...\n")
            self.prior_early_learner.fit(x_prior, y_prior, x_post, y_post)

            model = self.prior_learner.model
            for key, param in model.post_param_dict.items():
                if(isinstance(param, torch.nn.ParameterList)
                   or isinstance(param, list)):
                    param_list = param
                    for i in range(len(param_list)):
                        model.prior_param_dict[key][i].data = (
                            param_list[i].data.clone())
                else:
                    model.prior_param_dict[key].data = param.data.clone()

        if(self.step is None or self.step == "post"):

            self.__post_loss.m = len(X)-size_prior
            if(self.step == "post"):
                self.post_learner.load(self._load)

            logging.info("Running posterior learning ...\n")
            self.post_learner.fit(x_post, y_post)

        if(self.writer is not None):
            self.writer.write("state_dict", self.save())

    def save(self):
        if(self.step == "prior"):
            return self.prior_learner.save()
        else:
            return self.post_learner.save()

    def load(self, load_dict):
        if(self.step == "prior"):
            return self.prior_learner.load(load_dict)
        else:
            return self.post_learner.load(load_dict)

    def predict(self, X, init_keep=True):
        if(self.step == "prior"):
            model = self.prior_learner.model
        else:
            model = self.post_learner.model

        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        pred = None
        for i, batch in enumerate(loader):

            batch["x"] = batch["x"].to(
                device=self.device, dtype=torch.float32)

            batch["keep"] = True
            if(i == 0):
                batch["keep"] = init_keep

            model(batch)
            if(pred is None):
                pred = model.pred.cpu().detach().numpy()
            else:
                pred = np.concatenate(
                    (pred, model.pred.cpu().detach().numpy()))
        return pred[:, 0]

    def output(self, X, init_keep=True):
        if(self.step == "prior"):
            model = self.prior_learner.model
        else:
            model = self.post_learner.model

        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        output = None
        for i, batch in enumerate(loader):

            batch["keep"] = True
            if(i == 0):
                batch["keep"] = init_keep

            batch["x"] = batch["x"].to(
                device=self.device, dtype=torch.float32)
            model(batch)
            if(output is None):
                output = model.out.cpu().detach().numpy()
            else:
                output = np.concatenate(
                    (output, model.out.cpu().detach().numpy()))
        return output[:, 0]

###############################################################################
