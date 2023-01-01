import torch
import importlib
import inspect
import numpy as np
from core.kl_inv import klInvFunction
from core.C_param import CParamFunction
from core.k_param import kParamFunction
import math


###############################################################################

class MetaModules(type):

    def __get_class_dict(cls):
        class_dict = {}
        for class_name, class_ in inspect.getmembers(
            importlib.import_module("core.modules"), inspect.isclass
        ):
            if(class_name != "MetaModules" and class_name != "Modules"):
                class_name = class_name.replace("Modules", "")
                class_dict[class_name] = class_
        return class_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, torch.nn.Module, )

        # Getting the name of the module
        if("name" not in kwargs):
            class_name = args[0]
        else:
            class_name = kwargs["name"]

        # Getting the module dictionnary
        class_dict = cls.__get_class_dict()

        # Checking that the module exists
        if(class_name not in class_dict):
            raise Exception(class_name+" doesn't exist")

        # Adding the new module in the base classes
        bases = (class_dict[class_name], )+bases

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaModules, new_cls).__call__(*args, **kwargs)


# --------------------------------------------------------------------------- #


class Modules(metaclass=MetaModules):

    def __init__(self, name, model):
        super().__init__()
        self.model = model
        self.param = None

    def numpy_to_torch(self, x, y):
        if(isinstance(x, np.ndarray)):
            x = torch.tensor(x)
        if(isinstance(y, np.ndarray)):
            y = torch.tensor(y)
        return x, y

    def torch_to_numpy(self, x, y, m):
        # Note: m is consider as tensor
        if(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            m = m.detach().numpy()
        return m

    def float_to_numpy_torch(self, x, y, m):
        # Note: m is consider as tensor
        if(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            m = np.array(m)
        elif(isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            m = torch.tensor(m)
        return m

    def load(self, param):
        if(isinstance(param, torch.Tensor)):
            self.param.data = param.data

    def save(self):
        return self.param

    def fit(self, x, y):
        raise NotImplementedError


# --------------------------------------------------------------------------- #


class BoundedCrossEntropyLossModules():

    def __init__(self, name, model, L_max=4.0):
        super().__init__(name, model)
        self.L_max = L_max

    def fit(self, x, y):
        # Computing bounded cross entropy (from Dziugaite et al., 2018)
        x, y = self.numpy_to_torch(x, y)

        exp_L_max = torch.exp(-torch.tensor(self.L_max, requires_grad=False))
        #  x_ = torch.nn.functional.softmax(x, dim=1)
        x_ = exp_L_max + (1.0-2.0*exp_L_max)*x
        x_ = (1.0/self.L_max)*torch.log(x_)
        loss = torch.nn.functional.nll_loss(x_, y[:, 0])

        return self.torch_to_numpy(x, y, loss)


class ZeroOneLossModules():

    def fit(self, x, y):
        x, y = self.numpy_to_torch(x, y)
        loss = (x != y).float()
        loss = torch.mean(loss)
        return self.torch_to_numpy(x, y, loss)


class BoundModules():

    def __init__(
        self, name, model, risk, m, delta, T=1, sample=None, bound="ours"
    ):
        super().__init__(name, model)
        self.m = float(m)
        self.delta = delta
        self.T = T
        self.sample = None
        if(sample is not None):
            self.sample = float(sample)
        self._risk = risk
        self.bound = str(bound)
        self.device = next(model.parameters()).device
        self.c_ = torch.nn.Parameter(
            torch.tensor(math.inf, device=self.device), requires_grad=True)
        self.k_ = torch.nn.Parameter(
            torch.tensor(math.inf, device=self.device), requires_grad=True)
        self.__small = torch.tensor(1e-10, device=self.device)
        self.__zero = torch.tensor(0.0, device=self.device)

    def _bound_sample(self):
        return torch.tensor((1.0/self.sample)*(math.log(4.0/self.delta)))

    def _bound(self, div, delta=None):
        if(delta is None):
            delta = self.delta
        bound = (1.0/self.m)*(div+math.log(
            (self.T*2.0*math.sqrt(self.m))/delta))
        bound = torch.max(self.__small, bound)
        return bound

    def _bound_blanchard(self, div):
        nb_param = len(kParamFunction.list_k_param(self.m))
        k = kParamFunction.apply(self.k_, self.m)
        div = torch.max(self.__zero, div)
        bound = (1.0/self.m)*((1.0+(1.0/k))*div+torch.log(
            (self.T*nb_param*(k+1.0))/self.delta))
        bound = torch.max(self.__small, bound)
        return bound

    def _init_c(self, r, div):
        c_ = CParamFunction.list_c_param()
        bound_best = math.inf
        for i in range(len(c_)):
            self.c_.data = c_[i]
            b = self._bound_catoni(r, div)
            if(b < bound_best):
                bound_best = b
                c_best = c_[i]
        self.c_.data = c_best

    def _init_k(self):
        k_ = kParamFunction.list_k_param(self.m)
        self.k_.data = k_[0]

    def _bound_catoni(self, r, div):
        nb_param = len(CParamFunction.list_c_param())
        C = CParamFunction.apply(self.c_)
        m = self.m
        delta = self.delta
        term_1 = 1.0/(1.0-torch.exp(-C))
        term_2 = -C*r-(1.0/m)*div-(1.0/m)*math.log(self.T*nb_param/delta)
        return term_1*(1.0-torch.exp(term_2))

    def fit(self, x, y):
        x, y = self.numpy_to_torch(x, y)
        # NOTE: We assume that we already have the divergence
        r = self._risk(x, y)
        if(self.bound == "ours"):
            div = self.model.div_renyi
            b = klInvFunction.apply(
                r, self._bound(div, delta=(self.delta/2.0)**3.0), "MAX")
        elif(self.bound == "rivasplata"):
            div = self.model.div_rivasplata
            b = klInvFunction.apply(r, self._bound(div), "MAX")
        elif(self.bound == "catoni"):
            div = self.model.div_rivasplata
            if(torch.isinf(self.c_)):
                self._init_c(r, div)
            b = self._bound_catoni(r, div)
        elif(self.bound == "blanchard"):
            div = self.model.div_rivasplata
            if(torch.isinf(self.k_)):
                self._init_k()
            b = klInvFunction.apply(r, self._bound_blanchard(div), "MAX")
        elif(self.bound == "kl" or self.bound == "renyi"):
            div = self.model.div_kl
            if(self.bound == "renyi"):
                div = self.model.div_renyi
            if(self.sample is not None):
                r = klInvFunction.apply(r, self._bound_sample(), "MAX")
                b = klInvFunction.apply(r, self._bound(
                    div, delta=self.delta/2.0), "MAX")
            else:
                b = klInvFunction.apply(r, self._bound(
                    div, delta=self.delta), "MAX")

        return self.torch_to_numpy(x, y, b)
