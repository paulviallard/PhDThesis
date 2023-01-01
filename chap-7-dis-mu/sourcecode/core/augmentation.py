import numpy as np
import torch
from torchvision import transforms


class Augmentation():

    def __init__(self, fun_list):
        self.fun_list = fun_list
        self.transform = transforms.Compose(self.fun_list)

    def numpy_to_torch(self, x):
        if(isinstance(x, np.ndarray)):
            x = torch.tensor(x)
        return x

    def torch_to_numpy(self, x, x_):
        # Note: x_p is consider as tensor
        if(isinstance(x, np.ndarray)):
            x_ = x_.detach().cpu().numpy()
        return x_

    def fit(self, X):
        X_ = self.numpy_to_torch(X)
        X_ = self.transform(X_)
        return self.torch_to_numpy(X, X_)
