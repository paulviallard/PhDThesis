import torch

from betaincder import betainc, betaincderp, betaincderq

###############################################################################


def betaincderx(x, a, b):
    lbeta = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    partial_x = torch.exp((b-1)*torch.log1p(-x)+(a-1)*torch.log(x)-lbeta)
    return partial_x


class BetaInc(torch.autograd.Function):
    """ regularized incomplete beta function and
    its forward and backward passes"""

    @staticmethod
    def forward(ctx, p, q, x):

        x = torch.clamp(x, 0, 1)

        ctx.save_for_backward(p, q, x)
        # deal with dirac distributions
        if(p == 0.0):
            # for any x, cumulative = 1.
            return torch.tensor(1.0)

        elif(q == 0.0 or x == 0.0):
            # cumulative = 0.
            return torch.tensor(0.0)

        return torch.tensor(betainc(x, p, q))

    @staticmethod
    def backward(ctx, grad):
        p, q, x = ctx.saved_tensors

        if(p == 0. or q == 0. or x == 0.):
            # deal with dirac distributions
            grad_p = 0.0
            grad_q = 0.0
            grad_x = 0.0

        else:
            grad_p = betaincderp(x, p, q)
            grad_q = betaincderq(x, p, q)
            grad_x = betaincderx(x, p, q)

        return grad * grad_p, grad * grad_q, grad * grad_x


###############################################################################
