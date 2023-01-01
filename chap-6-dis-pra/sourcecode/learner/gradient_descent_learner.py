import torch
import numpy as np
import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from core.numpy_dataset import NumpyDataset


###############################################################################

class GradientDescentLearner(BaseEstimator, ClassifierMixin):

    def __init__(
        self, model, device,
        epoch=10, batch_size=None, writer=None
    ):
        self.model = model
        self.device = device

        self.epoch = epoch
        self.batch_size = batch_size
        self.writer = writer

    def fit(self, X, y):

        data = NumpyDataset({
            "x_train": X,
            "y_train": y})
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        # Computing batch size
        num_batch = int(len(data)/self.batch_size)
        if(len(data) % self.batch_size != 0):
            num_batch += 1

        for epoch in range(self.epoch):

            logging.info(("Running epoch [{}/{}] ...\n").format(
                epoch+1, self.epoch))

            loss_sum = 0.0
            crit_sum = 0.0

            for i, batch in enumerate(loader):

                batch["x"] = batch["x"].to(
                    device=self.device, dtype=torch.float32)
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.long).unsqueeze(0).T

                self._label = batch["y"]
                self._size = float(batch["size"][0])

                # Optimize the model
                # REQUIRED:
                #   (1) create self._loss
                #   (2) create the dict self._log
                self._loss = None
                self._log = {}
                self._write = {}
                self._optimize(batch)
                assert self._loss is not None

                # We write in the writer
                if(self.writer is not None and self._write != {}):
                    for key in self._write.keys():
                        self.writer.write(key, self._write[key])

                # Computing mean loss
                loss_sum += self._loss
                loss_mean = loss_sum/(i+1)

                # We print the loss and the other values
                logging_str = "[{}/{}] - loss {:.4f}".format(
                        i+1, num_batch, loss_mean)
                for key, value in self._log.items():
                    logging_str += self.__print_logging(key, value)
                logging.info(logging_str+"\r")

                if i+1 == num_batch:
                    logging.info("\n")

        return self

    def __print_logging(self, key, value):
        """
        Print the "value" associated to the "key"

        Parameters
        ----------
        key: str
            The name associated to the value
        value: int or float or str or tensor or ndarray
            The value to print
        """
        if(isinstance(value, int)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, float)):
            return " - {} {:.4f}".format(key, value)
        elif(isinstance(value, str)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, torch.Tensor)):
            return self.__print_logging(key, value.cpu().detach().numpy())
        elif(isinstance(value, np.ndarray)):
            if(value.ndim == 0):
                return self.__print_logging(key, value.item())
            else:
                raise ValueError("value cannot be an array")
        else:
            raise TypeError(
                "value must be of type torch.Tensor; np.ndarray;"
                + " int; float or str.")

    def output(self, X):
        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        output = None
        for i, batch in enumerate(loader):

            batch["x"] = batch["x"].to(
                device=self.device, dtype=torch.float32)
            self.model(batch)
            if(output is None):
                output = self.model.out.cpu().detach().numpy()
            else:
                output = np.concatenate(
                    (output, self.model.out.cpu().detach().numpy()))
        return output[:, 0]

    def predict(self, X):
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

            self.model(batch)
            if(pred is None):
                pred = self.model.pred.cpu().detach().numpy()
            else:
                pred = np.concatenate(
                    (pred, self.model.pred.cpu().detach().numpy()))
        return pred[:, 0]

    def predict_proba(self, X):
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

            self.model(batch)
            if(pred is None):
                pred = self.model.pred_proba.cpu().detach().numpy()
            else:
                pred = np.concatenate(
                    (pred, self.model.pred_proba.cpu().detach().numpy()))
        return pred

    def save(self):
        return self.model.state_dict()

    def load(self, state_dict):
        return self.model.load_state_dict(state_dict)

    def _optimize(self):
        raise NotImplementedError

###############################################################################
