import logging
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
from core.numpy_dataset import NumpyDataset

###############################################################################


class GradientDescentLearner(BaseEstimator, ClassifierMixin):

    def __init__(
        self, majority_vote, epoch=10, batch_size=None, writer=None
    ):
        assert isinstance(epoch, int) and epoch > 0
        self.epoch = epoch
        assert (batch_size is None
                or (isinstance(batch_size, int) and batch_size > 0))
        self.batch_size = batch_size

        self.device = "cpu"
        new_device = torch.device("cpu")
        if(torch.cuda.is_available() and self.device != "cpu"):
            new_device = torch.device(self.device)
        self.device = new_device

        self.writer = writer
        self.mv = majority_vote

    def fit(self, **kwargs):
        """
        Run a C-Bound minimization algorithm based on gradient descent

        Parameters
        ----------
        X: ndarray
            The inputs
        y: ndarray
            The labels
        """
        # X -> (size, nb_feature)
        # y -> (size, 1)
        #  assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        #  assert (len(X.shape) == 2 and len(y.shape) == 2 and
        #          X.shape[0] == y.shape[0] and
        #          y.shape[1] == 1 and X.shape[0] > 0)

        # We set the seed
        SEED = 0
        np.random.seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(SEED)

        # We initialize the "differentiable" majority vote
        #  self.mv_diff = self.mv

        # We initialize the dataset
        data_dict = {}
        for key, val in kwargs.items():
            data_dict[key+"_train"] = val
        data = NumpyDataset(data_dict)
        X = data_dict[list(data_dict.keys())[0]]

        if(self.batch_size is None or self.batch_size >= len(X)):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)
        num_batch = int(len(data)/self.batch_size)
        if(len(data) % self.batch_size != 0):
            num_batch += 1

        # For each epoch,
        self.epoch_ = 0
        while(self.epoch_ < self.epoch):

            if(self.batch_size != len(X)):
                logging.info(("Running epoch [{}/{}] ...\n").format(
                    self.epoch_+1, self.epoch))

            loss_sum = 0.0

            for i, batch in enumerate(loader):

                # We optimize the model
                # REQUIRED:
                #   (1) self._loss
                #   (2) the dict self._log
                # OPTIONAL:
                #   (1) the dict self._write
                self._loss = None
                self._log = {}
                self._write = {}
                self._optimize(batch)
                assert self._loss is not None

                # We write in the writer
                if(self.writer is not None and self._write != {}):
                    for key in self._write.keys():
                        self.writer.write(key, self._write[key])

                # We compute mean loss
                loss_sum += self._loss
                loss_mean = loss_sum/(i+1)

                if(self.batch_size != len(X)):
                    loading = (i+1, num_batch)
                else:
                    loading = (self.epoch_+1, self.epoch)

                # We print the loss and the other values
                logging_str = "[{}/{}] - loss {:.4f}".format(
                        loading[0], loading[1], loss_mean)
                for key, value in self._log.items():
                    logging_str += self.__print_logging(key, value)
                logging.info(logging_str+"\r")

                if(self.batch_size != len(X)):
                    if i+1 == num_batch:
                        logging.info("\n")

            self.epoch_ += 1

        if(self.batch_size == len(X)):
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

    def predict(self, X):
        """
        Predict the label of the inputs

        X: ndarray
            The inputs
        """

        # We get the "test" set
        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        # We get the "normal" posterior
        if(self.quasi_uniform):
            self.mv.switch_complemented()

        # We predict the inputs batch by batch
        pred = None
        for i, batch in enumerate(loader):
            batch["m"] = len(X)
            batch["x"] = batch["x"].to(
                device=self.device, dtype=torch.float32)

            self.mv_diff(batch)
            pred_ = self.mv_diff.pred.detach().numpy()

            if(pred is None):
                pred = pred_
            else:
                pred = np.concatenate((pred, pred_))

        # We switch to the "quasi-uniform" posterior (if wanted)
        if(self.quasi_uniform):
            self.mv.switch_complemented()

        return pred

    def _optimize(self):
        raise NotImplementedError


###############################################################################
