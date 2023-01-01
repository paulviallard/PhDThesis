import numpy as np
import torch
import cvxpy as cp
import math

###############################################################################

import warnings
import importlib
import inspect

from core.kl_inv import klInvFunction
from core.beta_inc import BetaInc


###############################################################################

class MetaModules(type):

    def __get_class_dict(cls):
        class_dict = {}
        for class_name, class_ in inspect.getmembers(
            importlib.import_module("core.modules"), inspect.isclass
        ):
            if(class_name != "MetaModules" and class_name != "Modules"):
                #  class_name = class_name.lower()
                class_name = class_name.replace("Modules", "")
                class_dict[class_name] = class_
        return class_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, )

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

    def __init__(self, name, majority_vote=None):
        super().__init__()
        self.mv = majority_vote

    def numpy_to_torch(self, *var_list):
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(var_list[i], np.ndarray)):
                new_var_list.append(torch.tensor(var_list[i]))
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

    def torch_to_numpy(self, ref, *var_list):
        # Note: elements in var_list are considered as tensor
        new_var_list = []
        for i in range(len(var_list)):
            if(isinstance(ref, np.ndarray)
               and isinstance(var_list[i], torch.Tensor)):
                new_var_list.append(var_list[i].detach().numpy())
            else:
                new_var_list.append(var_list[i])
        if(len(new_var_list) == 1):
            return new_var_list[0]
        return tuple(new_var_list)

    def fit(self, y, y_p):
        raise NotImplementedError


###############################################################################

class RiskModules():

    def fit(self, **kwargs):

        if("margin" in kwargs):
            margin = kwargs["margin"]
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif("x" in kwargs and "y" in kwargs):
            x = kwargs["x"]
            y = kwargs["y"]
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin

        elif("x_1" in kwargs and "y_1" in kwargs):
            i = 1
            batch = {}
            ref = kwargs[f"x_1"]
            while(f"x_{i}" in kwargs):
                batch[f"x_{i}"] = self.numpy_to_torch(kwargs[f"x_{i}"])
                batch[f"y_{i}"] = self.numpy_to_torch(kwargs[f"y_{i}"])
                i += 1
            self.mv(batch)
            margin = self.mv.margin

        else:
            raise RuntimeError("")

        # We compute the Gibbs Risk
        risk = torch.mean(0.5*(1.0-margin))
        return self.torch_to_numpy(ref, risk)


class DisagreementModules():

    def fit(self, x=None, y=None, margin=None):

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin

        else:
            raise RuntimeError("x, y must be None or margin must be None")

        # We compute the Disagreement
        disa = torch.mean(0.5*(1.0-margin**2.0))
        return self.torch_to_numpy(ref, disa)


class JointModules():

    def fit(self, x=None, y=None, margin=None):

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin

        else:
            raise RuntimeError("x, y must be None or margin must be None")

        risk = torch.mean(0.5*(1.0-margin))
        disa = torch.mean(0.5*(1.0-margin**2.0))
        joint = risk - 0.5*disa

        # We compute the Joint Error
        return self.torch_to_numpy(ref, joint)


class ZeroOneModules():

    def fit(self, y=None, pred=None, x=None):

        if(x is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            batch = {"x": x, "y": y}
            self.mv(batch)
            pred = self.mv.pred
        else:
            ref = y
            y, pred = self.numpy_to_torch(y, pred)

        assert (len(y.shape) == 2 and y.shape[1] == 1)
        assert (len(pred.shape) == 2 and pred.shape[1] == 1)
        zero_one = (y != pred).float()
        zero_one = torch.mean(zero_one)

        return self.torch_to_numpy(ref, zero_one)

# --------------------------------------------------------------------------- #


class McAllesterBoundModules():

    def fit(self, r, kl, m, delta, mode="MAX"):
        ref = r
        r, kl = self.numpy_to_torch(r, kl)

        assert (isinstance(m, int) or isinstance(m, float))
        assert (isinstance(delta, float))

        bound = (1.0/m)*(kl+np.log((2.0*np.sqrt(m))/delta))
        if(mode == "MAX"):
            bound = r + torch.sqrt(0.5*bound)
        if(mode == "MIN"):
            bound = r - torch.sqrt(0.5*bound)

        return self.torch_to_numpy(ref, bound)


class SeegerBoundModules():

    def fit(self, r, kl, m, delta, mode="MAX"):
        ref = r
        r, kl = self.numpy_to_torch(r, kl)

        assert (isinstance(m, int) or isinstance(m, float))
        assert (isinstance(delta, float))

        bound = (1.0/m)*(kl+np.log((2.0*np.sqrt(m))/delta))
        bound = klInvFunction.apply(r, bound, mode)

        return self.torch_to_numpy(ref, bound)

# --------------------------------------------------------------------------- #


class CBoundModules():

    def fit(self, r=None, d=None):

        ref = r
        r, d = self.numpy_to_torch(r, d)
        bound = self.__c_bound(r, d)

        return self.torch_to_numpy(ref, bound)

    def __c_bound(self, r, d):
        """
        Compute the C-Bound (The "third form" in [1])

        Parameters
        ----------
        r: tensor
            The risk
        d: tensor
            The disagreement
        """
        r = torch.min(torch.tensor(0.5).to(r.device), r)
        d = torch.max(torch.tensor(0.0).to(d.device), d)
        cb = (1.0-((1.0-2.0*r)**2.0)/(1.0-2.0*d))
        if(torch.isnan(cb) or torch.isinf(cb)):
            cb = torch.tensor(1.0, requires_grad=True)
        return cb


class CBoundJointModules():

    def fit(self, e=None, d=None):

        ref = e
        e, d = self.numpy_to_torch(e, d)
        bound = self.__c_bound(e, d)

        return self.torch_to_numpy(ref, bound)

    def __c_bound(self, e, d):
        """
        Compute the C-Bound of PAC-Bound 2 (see page 820 of [1])

        Parameters
        ----------
        e: ndarray
            The joint error
        d: ndarray
            The disagreement
        """
        return (1.0-((1.0-(2.0*e+d))**2.0)/(1.0-2.0*d))


class CBoundMcAllesterModules():

    def __init__(self, name, majority_vote=None, m=1, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta
        self.m = m
        self.__mcallester_bound = Modules("McAllesterBound", self.mv).fit
        self.__c_bound = Modules("CBound", self.mv).fit

    def fit(self, x=None, y=None, margin=None, rS=None, dS=None):

        assert((x is not None and y is not None)
               or (margin is not None)
               or (rS is not None and dS is not None))

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin
        else:
            ref = rS
            self.rS, self.dS = self.numpy_to_torch(rS, dS)

        if(margin is not None or (x is not None and y is not None)):
            # We compute the empirical risk and disagreement
            self.rS = Modules("Risk").fit(x=x, y=y, margin=margin)
            self.dS = Modules("Disagreement").fit(x=x, y=y, margin=margin)

        # We compute the PAC-Bayesian bounds for the risk and the disagreement
        self.rD = self.__mcallester_bound(
            self.rS, self.mv.kl, self.m, self.delta*0.5)
        self.dD = self.__mcallester_bound(
            self.dS, 2.0*self.mv.kl, self.m, self.delta*0.5, mode="MIN")
        if(self.rD > 0.5):
            return self.torch_to_numpy(ref, torch.tensor(1.0))

        # We compute the C-Bound with the PAC-Bayesian bounds
        bound = self.__c_bound(self.rD, self.dD)
        return self.torch_to_numpy(ref, bound)


class CBoundSeegerModules():

    def __init__(self, name, majority_vote=None, m=1, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta
        self.m = m
        self.__seeger_bound = Modules("SeegerBound", self.mv).fit
        self.__c_bound = Modules("CBound", self.mv).fit

    def fit(self, x=None, y=None, margin=None, rS=None, dS=None):

        assert((x is not None and y is not None)
               or (margin is not None)
               or (rS is not None and dS is not None))

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin
        else:
            ref = rS
            self.rS, self.dS = self.numpy_to_torch(rS, dS)

        if(margin is not None or (x is not None and y is not None)):
            # We compute the empirical risk and disagreement
            self.rS = Modules("Risk").fit(x=x, y=y, margin=margin)
            self.dS = Modules("Disagreement").fit(x=x, y=y, margin=margin)

        # We compute the PAC-Bayesian bounds for the risk and the disagreement
        self.rD = self.__seeger_bound(
            self.rS, self.mv.kl, self.m, self.delta*0.5)
        self.dD = self.__seeger_bound(
            self.dS, 2.0*self.mv.kl, self.m, self.delta*0.5, mode="MIN")
        if(self.rD > 0.5):
            return self.torch_to_numpy(ref, torch.tensor(1.0))

        # We compute the C-Bound with the PAC-Bayesian bounds
        bound = self.__c_bound(self.rD, self.dD)
        return self.torch_to_numpy(ref, bound)


class CBoundLacasseModules():

    def __init__(self, name, majority_vote=None, m=1, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta
        self.m = m
        self.__c_bound = Modules("CBoundJoint", self.mv).fit

    def fit(self, x=None, y=None, margin=None, eS=None, dS=None):

        assert((x is not None and y is not None)
               or (margin is not None)
               or (eS is not None and dS is not None))

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin
        else:
            ref = eS
            self.eS, self.dS = self.numpy_to_torch(eS, dS)

        if(margin is not None or (x is not None and y is not None)):
            # We compute the empirical risk and disagreement
            self.eS = Modules("Joint").fit(x=x, y=y, margin=margin)
            self.dS = Modules("Disagreement").fit(x=x, y=y, margin=margin)

        # We compute the PAC-Bayesian bounds for the risk and the disagreement

        if(2.0*self.eS+self.dS >= 1.0
           or self.dS > 2*(torch.sqrt(self.eS)-self.eS)):
            self.eD = self.eS.item()
            self.dD = self.dS.item()
            return self.torch_to_numpy(ref, torch.tensor(1.0))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            (self.eD, self.dD) = self.__joint_disagreement_bound(
                self.eS, self.dS,
                self.mv.kl, self.m)

        # We compute the C-Bound with the PAC-Bayesian bounds
        bound = self.__c_bound(self.eD, self.dD)

        return self.torch_to_numpy(ref, bound)

    def __bound(self, kl, m):
        """
        Compute the PAC-Bayesian bound of PAC-Bound 2 (see page 820 of [1])

        Parameters
        ----------
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        """
        b = np.log((2.0*np.sqrt(m)+m)/(self.delta))
        b = (1.0/m)*(2.0*kl+b)
        return b

    def __joint_disagreement_bound(self, eS, dS, kl, m, tol=0.01):
        """
        Solve the inner maximization problem using the
        "Bisection method for quasiconvex optimization" of [3] (p 146)

        Parameters
        ----------
        eS: ndarray
            The empirical joint error
        dS: ndarray
            The empirical disagreement
        kl: ndarray
            The KL divergence
        m: float
            The number of data
        tol: float, optional
            The tolerance parameter
        """
        u = 1.0
        l = 0.0

        bound = self.__bound(kl, m).item()
        eS = eS.item()
        dS = dS.item()

        # For numerical stability
        eS = np.maximum(eS, 0.0)
        dS = np.maximum(dS, 0.0)

        while(u-l > tol):
            t = (l+u)/2.0

            e = cp.Variable(shape=1, nonneg=True)
            d = cp.Variable(shape=1, nonneg=True)
            e_min = cp.atoms.affine.hstack.hstack([e, 0.25])
            prob = cp.Problem(
                cp.Minimize((1-(2*e+d))**2.0-t*(1-2*d)),
                [(cp.kl_div(eS, e)+cp.kl_div(dS, d)
                  + cp.kl_div((1-eS-dS), 1-e-d) <= bound),
                 d <= 2.0*(cp.sqrt(cp.atoms.min(e_min))-e)])

            prob.solve()

            if(e.value is None or d.value is None):
                # Only in case where the solution is not found
                return (None, None)
            else:
                e = e.value[0]
                d = d.value[0]

            c_bound = 1.0-((1-(2*e+d))**2.0)/(1-2*d)

            if(c_bound > 1.0-t):
                u = t
            else:
                l = t

        return (e, d)


class BoundRiskModules():

    def __init__(self, name, majority_vote=None, m=1, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta
        self.m = m
        self.__seeger_bound = Modules("SeegerBound", self.mv).fit

    def fit(self, x=None, y=None, margin=None, rS=None):

        assert((x is not None and y is not None)
               or (margin is not None)
               or (rS is not None))

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin
        else:
            ref = rS
            self.rS = rS

        if(margin is not None or (x is not None and y is not None)):
            # We compute the empirical risk and disagreement
            self.rS = Modules("Risk").fit(x=x, y=y, margin=margin)

        # We compute the PAC-Bayesian bounds for the risk and the disagreement
        self.rD = self.__seeger_bound(
            self.rS, self.mv.kl, self.m, self.delta)

        # We compute the C-Bound with the PAC-Bayesian bounds
        bound = 2.0*self.rD
        return self.torch_to_numpy(ref, bound)


class BoundJointModules():

    def __init__(self, name, majority_vote=None, m=1, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta
        self.m = m
        self.__seeger_bound = Modules("SeegerBound", self.mv).fit

    def fit(self, x=None, y=None, margin=None, eS=None):

        assert((x is not None and y is not None)
               or (margin is not None)
               or (eS is not None))

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin
        else:
            ref = eS
            self.eS = eS

        if(margin is not None or (x is not None and y is not None)):
            # We compute the empirical risk and disagreement
            self.eS = Modules("Joint").fit(x=x, y=y, margin=margin)

        # We compute the PAC-Bayesian bounds for the risk and the disagreement
        self.eD = self.__seeger_bound(
            self.eS, 2.0*self.mv.kl, self.m, self.delta)

        # We compute the C-Bound with the PAC-Bayesian bounds
        bound = 4.0*self.eD
        return self.torch_to_numpy(ref, bound)

###############################################################################


class BoundStoModules():

    def __init__(self, name, majority_vote=None, m=1, delta=0.05):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta
        self.m = m
        self.__seeger_bound = Modules("SeegerBound", self.mv).fit

    def fit(self, **kwargs):

        if("margin" in kwargs):
            margin = kwargs["margin"]
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif("x" in kwargs and "y" in kwargs):
            x = kwargs["x"]
            y = kwargs["y"]
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin

        elif("x_1" in kwargs):
            i = 1
            batch = {}
            ref = kwargs[f"x_1"]
            while(f"x_{i}" in kwargs):
                batch[f"x_{i}"] = self.numpy_to_torch(kwargs[f"x_{i}"])
                batch[f"y_{i}"] = self.numpy_to_torch(kwargs[f"y_{i}"])
                i += 1
            self.mv(batch)
            margin = self.mv.margin

        else:
            raise RuntimeError("")

        self.sto_risk = Modules("Risk").fit(margin=margin)

        # We compute the PAC-Bayesian bounds for the stochastic risk
        self.sto_true_risk = self.__seeger_bound(
            self.sto_risk, self.mv.kl, self.m, self.delta)

        # We compute the PAC-Bayesian bounds
        return self.torch_to_numpy(ref, self.sto_true_risk)

# --------------------------------------------------------------------------- #


class BoundRandModules():

    def __init__(self, name, majority_vote=None, m=1, delta=0.05, rand_n=100):
        super().__init__(name, majority_vote=majority_vote)
        self.delta = delta
        self.m = m
        self.rand_n = rand_n
        self.__seeger_bound = Modules("SeegerBound", self.mv).fit

    def fit(self, x=None, y=None, margin=None, rand_risk=None):

        assert((x is not None and y is not None)
               or (margin is not None)
               or (rand_risk is not None))

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)

        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)

            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin
        else:
            ref = rand_risk
            self.rand_risk = rand_risk

        if(margin is not None or (x is not None and y is not None)):
            # We compute the empirical risk
            self.rand_risk = Modules("Rand").fit(
                x=x, y=y, margin=margin, rand_n=self.rand_n)

        # We compute the PAC-Bayesian bounds
        self.rand_true_risk = self.__seeger_bound(
            self.rand_risk, self.rand_n*self.mv.kl, self.m, self.delta)

        # We compute the PAC-Bayesian bound
        bound = 2.0*self.rand_true_risk
        return self.torch_to_numpy(ref, bound)


class RandModules():

    def fit(self, x=None, y=None, margin=None, rand_n=100):

        if(margin is not None):
            ref = margin
            margin = self.numpy_to_torch(margin)
            assert (len(margin.shape) == 2 and margin.shape[1] == 1)
        elif(x is not None and y is not None):
            ref = x
            x, y = self.numpy_to_torch(x, y)
            assert (len(y.shape) == 2 and len(x.shape) == 2 and
                    y.shape[0] == x.shape[0] and
                    y.shape[1] == 1)
            batch = {"x": x, "y": y}
            self.mv(batch)
            margin = self.mv.margin
        else:
            raise RuntimeError("x, y must be None or margin must be None")

        risk = (0.5*(1.0-margin)).squeeze(1)
        rand_k = math.ceil(rand_n/2)
        risk = torch.stack([
            BetaInc.apply(torch.tensor(rand_k),
                          torch.tensor(rand_n-rand_k+1), r) for r in risk])
        risk = torch.mean(risk)
        return self.torch_to_numpy(ref, risk)


###############################################################################
