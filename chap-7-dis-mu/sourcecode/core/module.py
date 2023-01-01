import torch
import importlib
import inspect
import numpy as np
import math
from core.kl_inv import kl_inv

###############################################################################


class MetaModule(type):

    def __get_class_dict(cls):
        class_dict = {}
        for class_name, class_ in inspect.getmembers(
            importlib.import_module("core.module"), inspect.isclass
        ):
            if(class_name != "MetaMetrics" and class_name != "Module"):
                class_name = class_name.replace("Module", "")
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
        return super(MetaModule, new_cls).__call__(*args, **kwargs)


# --------------------------------------------------------------------------- #


class Module(metaclass=MetaModule):

    def __init__(self, name, model):
        super().__init__()
        self.model = model
        self.param = None

    def numpy_to_torch(self, x, y):
        if(isinstance(x, np.ndarray)):
            x = torch.tensor(x, device=self.model.device)
        if(isinstance(y, np.ndarray)):
            y = torch.tensor(y, device=self.model.device)
        return x, y

    def torch_to_numpy(self, x, y, m):
        # Note: m is consider as tensor
        if(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            m = m.detach().cpu().numpy()
        return m

    def float_to_numpy_torch(self, x, y, m):
        # Note: m is consider as tensor
        if(isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
            m = np.array(m)
        elif(isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            m = torch.tensor(m, device=x.device)
        return m

    def load(self, param):
        if(isinstance(param, torch.Tensor)):
            self.param.data = param.data

    def save(self):
        return self.param

    def fit(self, x, y):
        raise NotImplementedError


# --------------------------------------------------------------------------- #

class BoundedCrossEntropyModule():

    def __init__(self, name, model, L_max=4.0):
        super().__init__(name, model)
        self.L_max = L_max

    def fit(self, x, y):
        # Computing bounded cross entropy (from Dziugaite et al., 2018)
        x, y = self.numpy_to_torch(x, y)

        exp_L_max = torch.exp(-torch.tensor(self.L_max, requires_grad=False))
        x_ = exp_L_max + (1.0-2.0*exp_L_max)*x
        x_ = (1.0/self.L_max)*torch.log(x_)
        loss = torch.nn.functional.nll_loss(x_, y[:, 0])

        return self.torch_to_numpy(x, y, loss)


class ZeroOneModule():

    def fit(self, out, y):
        out, y = self.numpy_to_torch(out, y)

        if(out.shape[1] == 1):
            pred = out
        else:
            _, pred = torch.max(out, axis=1)
            pred = pred.unsqueeze(1)

        loss = torch.mean((pred != y).float())
        return self.torch_to_numpy(out, y, loss)


class ObjectiveModule():

    def __init__(self, name, model, alpha=1.0):

        super().__init__(name, model)
        self.__alpha = float(alpha)
        self.__risk = Module("ZeroOne", model).fit
        self.__loss = Module("BoundedCrossEntropy", model).fit

    def fit(self, x, y):
        x, y = self.numpy_to_torch(x, y)

        loss = self.__loss(x, y)
        risk = self.__risk(x, y)
        m = self.model.kwargs["measure"]
        measure = self.model.measures[m]

        loss_ = loss + (1.0/self.__alpha)*measure
        risk_ = risk + (1.0/self.__alpha)*measure

        objective = torch.cat(
            (loss_.unsqueeze(0), risk_.unsqueeze(0),
             loss.unsqueeze(0), risk.unsqueeze(0),
             measure.unsqueeze(0)), axis=0)
        return self.torch_to_numpy(x, y, objective)


class EmpRiskBoundModule():

    def __init__(self, name, model):

        super().__init__(name, model)
        self.__risk = Module("ZeroOne", model).fit

    def fit(self, pred, y, bound, bound_type="mcallester"):

        risk = self.__risk(pred, y)
        if(bound_type == "mcallester"):
            return risk + np.sqrt(0.5*bound.item())

        elif(bound_type == "seeger"):
            return kl_inv(risk.item(), bound.item(), "MAX")

        else:
            raise ValueError("bound_type must be mcallester or seeger")


class BoundModule():

    def __init__(self, name, model, learner, m, delta, alpha=1.0):

        super().__init__(name, model)
        self.m = m
        self.delta = delta
        self.zero_tensor = torch.tensor(0.0, device=self.model.device)
        self.__alpha = float(alpha)
        self.__risk = Module("ZeroOne", model).fit
        self.__loss = Module("BoundedCrossEntropy", model).fit

        self.learner = learner

    def fit(self, x, pred, out, y):

        m_ = self.model.kwargs["measure"]
        measure = self.model.measures[m_]

        risk = self.__risk(pred, y)
        post_objective = self.__alpha*risk + measure

        self.model.initialize()
        pred_ = self.learner.predict(x)
        out_ = self.learner.output(x)

        batch = {
            "out": out_,
            "pred": pred_,
            "x": x,
            "y": y,
            "size": len(x),
            "step": "measure",
            "alpha": self.__alpha
        }
        self.model.get_measures(batch)
        risk_ = self.__risk(pred_, y)
        measure_ = self.model.measures[m_]
        prior_objective = self.__alpha*risk_ + measure_

        bound = prior_objective - post_objective
        bound = (1/self.m)*(
            bound + math.log((2*math.sqrt(self.m))/((0.5*self.delta)**2.0)))
        return self.torch_to_numpy(x, y, bound)
