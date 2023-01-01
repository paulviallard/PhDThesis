import math
import torch
from torch.optim import Optimizer

###############################################################################


class MH(Optimizer):

    def __init__(
        self, params, weight_decay=0, lr=0.1, alpha=1.0, do_sgd=False
    ):
        """
        Initialize the (Stochastic) MALA optimizer

        Parameters
        ----------
        params :
            Parameters of the model to optimize
        weight_decay : float
            The weight decay applied to the parameters
        """
        defaults = dict(weight_decay=weight_decay)
        super(MH, self).__init__(params, defaults)

        assert weight_decay >= 0.0
        assert lr > 0.0
        assert alpha > 0.0
        self.weight_decay = weight_decay
        self._alpha = alpha
        self._lr = lr
        self._sigma = math.sqrt(2.0*self._lr*(1.0/self._alpha))
        self._do_sgd = do_sgd
        self.state = {}

    def set_lr(self, lr):
        self._lr = lr
        self._sigma = math.sqrt(2.0*self._lr*(1.0/self._alpha))

    def set_do_sgd(self, do_sgd):
        self._do_sgd = do_sgd

    def step(self, loss, risk, closure=None):
        # We update all the parameters with the Metropolis-Hastings algorithm
        # (see [1] for an introduction)
        for group in self.param_groups:
            for w in group['params']:

                # We get the gradient
                grad = w.grad

                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "MH does not support sparse gradients")

                # We add the weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(w, alpha=group['weight_decay'])

                # We initialize the state
                if(w not in self.state):
                    self.state[w] = {}
                state = self.state[w]

                # We generate x' \sim {\cal N}(w-\alpha*\nabla\ell, \sigma^2)
                state["w_old"] = w.data

                state["w_old_grad"] = state["w_old"]-self._lr*grad

                if(not(self._do_sgd)):
                    state["w_noise"] = torch.randn(
                        w.shape, device=state["w_old_grad"].device)*self._sigma
                    state["w_new"] = state["w_old_grad"] + state["w_noise"]
                else:
                    state["w_new"] = state["w_old_grad"]

                w.data = state["w_new"]

        if(self._do_sgd):
            return loss

        if closure is None:
            raise RuntimeError("The closure function is not available")
        new_loss, new_risk, _, _, _ = closure()

        w_old = None
        w_new = None
        w_old_grad = None
        w_new_grad = None

        for group in self.param_groups:
            for w in group['params']:

                # We get the gradient
                grad = w.grad

                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "MH does not support sparse gradients")

                # We add the weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(w, alpha=group['weight_decay'])

                state = self.state[w]
                state["w_new_grad"] = state["w_new"] - self._lr*grad

                if(w_old is None):
                    w_old = torch.flatten(state["w_old"])
                    w_new = torch.flatten(state["w_new"])
                    w_old_grad = torch.flatten(state["w_old_grad"])
                    w_new_grad = torch.flatten(state["w_new_grad"])
                else:
                    w_old = torch.cat(
                        (w_old, torch.flatten(state["w_old"])), axis=0)
                    w_new = torch.cat(
                        (w_new, torch.flatten(state["w_new"])), axis=0)
                    w_old_grad = torch.cat(
                        (w_old_grad,
                         torch.flatten(state["w_old_grad"])), axis=0)
                    w_new_grad = torch.cat(
                        (w_new_grad,
                         torch.flatten(state["w_new_grad"])), axis=0)

        density_candidate_old_new = -(1.0/(2.0*(self._sigma**2.0)))*(
            torch.norm(w_old-w_new_grad)**2.0)
        density_candidate_new_old = -(1.0/(2.0*(self._sigma**2.0)))*(
            torch.norm(w_new-w_old_grad)**2.0)

        density_target_new = -new_risk
        density_target_old = -risk

        alpha_prob = (density_target_new + density_candidate_old_new
                      - density_target_old - density_candidate_new_old)

        alpha_prob = torch.minimum(
            torch.tensor(1.0, device=alpha_prob.device), torch.exp(alpha_prob))

        # We generate a uniform noise u \sim {\cal U}(0, 1)
        u = torch.rand(1, device=alpha_prob.device)
        if(u > alpha_prob):
            for group in self.param_groups:
                for w in group['params']:
                    state = self.state[w]
                    w.data = state["w_old"]

        return loss

###############################################################################

# References:
# [1] Understanding the Metropolis-Hastings Algorithm
#     Siddhartha Chib, Edward Greenberg, 1995
