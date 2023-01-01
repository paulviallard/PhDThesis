import torch
import math


class CParamFunction(torch.autograd.Function):

    @staticmethod
    def list_c_param(a=-3.0, b=3.0):
        c = torch.arange(a, b+1.0, dtype=float)
        c[0] = c[0]+0.1
        c[-1] = c[-1]-0.1
        c_ = torch.log((-a+c)/(b-c))
        return c_

    @staticmethod
    def forward(ctx, c, a=-3.0, b=3.0):
        assert isinstance(c, torch.Tensor) and len(c.shape) == 0
        assert isinstance(a, float)
        assert isinstance(b, float) and b > a

        ctx.a = a
        ctx.b = b
        ctx.save_for_backward(c)
        return torch.pow(
            10.0, torch.round((b-a)*torch.sigmoid(c)+a))

    @staticmethod
    def backward(ctx, grad_output):
        c, = ctx.saved_tensors
        a, b = ctx.a, ctx.b

        c_ = torch.nn.Parameter(c.clone().detach(), requires_grad=True)
        with torch.autograd.enable_grad():
            test = torch.pow(10.0, (b-a)*torch.sigmoid(c_)+a)
            test.retain_grad()
            test.backward()
            grad_c = c_.grad

        return grad_output*grad_c
