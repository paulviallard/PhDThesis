import torch
import math


class kParamFunction(torch.autograd.Function):

    @staticmethod
    def list_k_param(m):
        max = int(math.sqrt(2.0*math.sqrt(m)-1.0))
        c = torch.arange(1.0, max+1.0, dtype=float)
        c[0] = c[0]+0.1
        c[-1] = c[-1]-0.1
        c_ = torch.log((-1.0+c)/(max-c))
        return c_

    @staticmethod
    def forward(ctx, c, m):
        assert isinstance(c, torch.Tensor) and len(c.shape) == 0
        max = int(math.sqrt(2.0*math.sqrt(m)-1.0))
        ctx.m = m
        ctx.save_for_backward(c)
        return torch.round((max-1.0)*torch.sigmoid(c)+1.0)

    @staticmethod
    def backward(ctx, grad_output):
        c, = ctx.saved_tensors
        m = ctx.m
        max = int(math.sqrt(2.0*math.sqrt(m)-1.0))

        c_ = torch.nn.Parameter(c.clone().detach(), requires_grad=True)
        with torch.autograd.enable_grad():
            test = (max-1.0)*torch.sigmoid(c_)+1.0
            test.retain_grad()
            test.backward()
            grad_c = c_.grad

        return grad_output*grad_c, None
