import torch
from torch.optim import Optimizer

###############################################################################


class iRProp(Optimizer):

    def __init__(self, params, weight_decay=0,
                 lr_init=0.1, lr_min=10**-20, lr_max=10**5,
                 inc_factor=1.1, dec_factor=0.5):
        """
        Initialize the iRProp optimizer

        Parameters
        ----------
        params :
            Parameters of the model to optimize
        weight_decay : float
            The weight decay applied to the parameters
        lr_init: float
            The initial learning rates (one learning rate per parameter)
        lr_min: float
            The lower-bound on the learning rates
        lr_max: float
            The upper-bound on the learning rates
        inc_factor: float
            The lr increasing factor (must be >= 1)
        dec_factor: float
            The lr decreasing factor (must be <= 1)
        """
        defaults = dict(weight_decay=weight_decay)
        super(iRProp, self).__init__(params, defaults)

        assert weight_decay >= 0.0
        self.inc_factor = inc_factor
        self.dec_factor = dec_factor

        self.lr_init = lr_init
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.t = 1
        self.state = {}

    def step(self, loss):
        # We update all the parameters with iRProp
        # that was introduced in [1]
        for group in self.param_groups:
            for w in group['params']:

                # We get the gradient
                grad = w.grad

                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError(
                        "iRProp does not support sparse gradients")

                # We initalize the state
                if(w not in self.state):
                    self.state[w] = {
                        "grad": [torch.zeros(w.shape)],
                        "lr": torch.ones(w.shape)*self.lr_init,
                        "w": [w.data],
                        "loss": [loss],
                    }
                state = self.state[w]

                #####################################################

                state["grad"].append(grad)
                state["w"].append(w.data)
                state["loss"].append(loss)

                # We update the lr size
                state["grad"][self.t]
                #  print(state["grad"][self.t-1], "[grad][self.t-1]")
                det = state["grad"][self.t]*state["grad"][self.t-1]
                # We increase the lr when det>0
                state["lr"][det > 0] = state["lr"][det > 0]*self.inc_factor
                # We increase the lr when det<0
                state["lr"][det < 0] = state["lr"][det < 0]*self.dec_factor
                # We clamp the lr with the min/max
                state["lr"] = torch.clamp(
                    state["lr"], self.lr_min, self.lr_max)

                # We update the parameters w
                # If det >= 0, it is the same as RProp
                state["w"][self.t][det >= 0] = (
                    state["w"][self.t-1][det >= 0]
                    - (state["lr"]*torch.sign(state["grad"][self.t])
                       )[det >= 0])

                delta = -1.0
                if self.t > 1:
                    delta = state["loss"][self.t-1]-state["loss"][self.t-2]

                # If func(x[t-1])>func(x[t-2]), we set x[t] to x[t-2]
                # when det<0 (only happens if t>1, as det==0 for t=1)
                if delta > 0:
                    state["w"][self.t][det < 0] = state["w"][self.t-2][det < 0]
                else:
                    state["w"][self.t][det < 0] = (
                        state["w"][self.t-1][det < 0]
                        - (state["lr"]*torch.sign(state["grad"][self.t])
                           )[det < 0])

                # We reset the gradient when det<0
                state["grad"][self.t][det < 0] = 0

                w.data = state["w"][self.t]

        self.t += 1

        return loss

###############################################################################

# References:
# [1] Empirical evaluation of the improved Rprop learning algorithm
#     Christian Igel and Michael Hüsken, 2003
