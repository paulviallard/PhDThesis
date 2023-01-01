import math
import torch

###############################################################################


def kl_inv(q, epsilon, mode, tol=10**-9, nb_iter_max=1000):
    """
    Solve the optimization problem min{ p in [0, 1] | kl(q||p) <= epsilon }
    or max{ p in [0,1] | kl(q||p) <= epsilon } for q and epsilon fixed
    Parameters
    ----------
    q: float
        The parameter q of the kl divergence
    epsilon: float
        The upper bound on the kl divergence
    tol: float, optional
        The precision tolerance of the solution
    nb_iter_max: int, optinal
        The maximum number of iterations
    """
    assert mode == "MIN" or mode == "MAX"
    assert isinstance(q, float) and q >= 0 and q <= 1
    assert isinstance(epsilon, float) and epsilon > 0.0

    def kl(q, p):
        """
        Compute the KL divergence between two Bernoulli distributions
        (denoted kl divergence)
        Parameters
        ----------
        q: float
            The parameter of the posterior Bernoulli distribution
        p: float
            The parameter of the prior Bernoulli distribution
        """
        return q*math.log(q/p)+(1-q)*math.log((1-q)/(1-p))

    # We optimize the problem with the bisection method

    if(mode == "MAX"):
        p_max = 1.0
        p_min = q
    else:
        p_max = q
        p_min = 10.0**-9

    for _ in range(nb_iter_max):
        p = (p_min+p_max)/2.0

        if(kl(q, p) == epsilon or (p_max-p_min)/2.0 < tol):
            return p

        if(mode == "MAX" and kl(q, p) > epsilon):
            p_max = p
        elif(mode == "MAX" and kl(q, p) < epsilon):
            p_min = p
        elif(mode == "MIN" and kl(q, p) > epsilon):
            p_min = p
        elif(mode == "MIN" and kl(q, p) < epsilon):
            p_max = p

    return p

###############################################################################


class klInvFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, epsilon, mode):
        assert mode == "MIN" or mode == "MAX"
        assert isinstance(q, torch.Tensor) and len(q.shape) == 0
        assert (isinstance(epsilon, torch.Tensor)
                and len(epsilon.shape) == 0 and epsilon > 0.0)
        ctx.save_for_backward(q, epsilon)

        # We solve the optimization problem to find the optimal p
        out = kl_inv(q.item(), epsilon.item(), mode)

        if(out < 0.0):
            out = 10.0**-9

        out = torch.tensor(out, device=q.device)
        ctx.out = out
        ctx.mode = mode
        return out

    @staticmethod
    def backward(ctx, grad_output):
        q, epsilon = ctx.saved_tensors
        grad_q = None
        grad_epsilon = None

        # We compute the gradient with respect to q and epsilon
        # (see [1])

        term_1 = (1.0-q)/(1.0-ctx.out)
        term_2 = (q/ctx.out)

        grad_q = torch.log(term_1/term_2)/(term_1-term_2)
        grad_epsilon = (1.0)/(term_1-term_2)

        return grad_output*grad_q, grad_output*grad_epsilon, None

###############################################################################

# References:
# [1] Learning Gaussian Processes by Minimizing PAC-Bayesian
#     Generalization Bounds
#     David Reeb, Andreas Doerr, Sebastian Gerwinn, Barbara Rakitsch, 2018
