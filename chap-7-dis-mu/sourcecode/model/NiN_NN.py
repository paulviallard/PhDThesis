# Based on https://github.com/nitarshan/robust-generalization-measures/
# blob/master/data/generation/models.py

import math
import torch


class Conv2dBatchNormBlock(torch.nn.Module):

    def __init__(
        self, device, in_channel, out_channel,
        kernel_size, stride=1, padding=0
    ):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        self.param = torch.nn.ParameterDict()

        self.weight = torch.nn.Parameter(
            torch.empty((
                self.out_channel, self.in_channel,
                self.kernel_size, self.kernel_size
            ), device=self.device, requires_grad=True))
        self.bias = torch.nn.Parameter(
            torch.empty((
                self.out_channel
            ), device=self.device, requires_grad=True))

        self.param["weight"] = self.weight
        self.param["bias"] = self.bias

        self.initialize()

    def initialize(self):
        # Initializing weights with Kaiming He initializer
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        gain = torch.nn.init.calculate_gain("leaky_relu")
        self.weight_bound = gain*math.sqrt(3.0/fan_in)
        torch.nn.init.uniform_(
            self.weight, -self.weight_bound, self.weight_bound)
        self.bias_bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -self.bias_bound, self.bias_bound)

    def forward(self, x, step):

        self.weight.data = torch.clamp(
            self.weight.data, -self.weight_bound, self.weight_bound)
        self.bias.data = torch.clamp(
            self.bias.data, -self.bias_bound, +self.bias_bound)

        x = torch.nn.functional.conv2d(
            x, self.weight, bias=self.bias,
            stride=self.stride, padding=self.padding)
        return x


class NiNBlock(torch.nn.Module):

    def __init__(self, in_channel, out_channel, device):
        super().__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel

        self.conv2d_batch_norm_block_list = []
        self.conv2d_batch_norm_block_list.append(Conv2dBatchNormBlock(
            device, self.in_channel, self.out_channel,
            3, stride=2, padding=1))
        self.conv2d_batch_norm_block_list.append(Conv2dBatchNormBlock(
            device, self.out_channel, self.out_channel,
            1, stride=1))
        self.conv2d_batch_norm_block_list.append(Conv2dBatchNormBlock(
            device, self.out_channel, self.out_channel,
            1, stride=1))

        self.conv2d_batch_norm_block_list = torch.nn.ModuleList(
            self.conv2d_batch_norm_block_list)

    def initialize(self):
        for block in self.conv2d_batch_norm_block_list:
            block.initialize()

    def forward(self, x, step):

        for block in self.conv2d_batch_norm_block_list:
            x = block(x, step)
            x = torch.nn.functional.leaky_relu(x)

        return x


class Model(torch.nn.Module):

    def __init__(self, device):

        super().__init__()

        self.device = device
        self.step = "train"

        self.depth = int(self.kwargs["depth"])
        self.width = int(self.kwargs["width"])
        self.input_size = self.kwargs["input_size"]
        self.class_size = int(self.kwargs["class_size"])

        # ------------------------------------------------------------------- #

        self.block_list = []
        self.block_list.append(NiNBlock(
            self.input_size[1], self.width, device))

        for i in range(self.depth-1):
            self.block_list.append(NiNBlock(
                self.width, self.width, device))
        self.block_list = torch.nn.ModuleList(self.block_list)

        # ------------------------------------------------------------------- #

        self.conv2d_batch_norm_block = Conv2dBatchNormBlock(
            device, self.width, self.class_size, 1, stride=1)

        # ------------------------------------------------------------------- #

        self.step = "measure"
        self._init_param = []
        for param in self.parameters():
            param = param.clone().detach()
            self._init_param.append(param)
        self.step = None

    def initialize(self):
        self.conv2d_batch_norm_block.initialize()
        for block in self.block_list:
            block.initialize()

    def forward(self, batch):

        x = batch["x"]
        old_step = self.step
        self.step = batch["step"]

        # --------------------------------------------------------------- #

        old_shape_1 = x.shape[2]
        old_shape_2 = x.shape[3]
        new_shape_1 = 2**math.ceil(math.log(x.shape[2], 2))
        new_shape_2 = 2**math.ceil(math.log(x.shape[3], 2))

        x = torch.nn.functional.pad(
            x, ((new_shape_2-old_shape_2)//2+(new_shape_2-old_shape_2) % 2,
                (new_shape_2-old_shape_2)//2,
                (new_shape_1-old_shape_1)//2+(new_shape_1-old_shape_1) % 2,
                (new_shape_1-old_shape_1)//2))

        for block in self.block_list:
            x = block(x, self.step)

        x = self.conv2d_batch_norm_block(x, self.step)

        x = torch.nn.functional.leaky_relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.squeeze(dim=2)
        x = x.squeeze(dim=2)

        # --------------------------------------------------------------- #

        self.raw_out = x
        self.out = torch.nn.functional.softmax(x, dim=1)
        self.pred = torch.max(x, dim=1)[1].unsqueeze(1)

        if(self.step == "train" or self.step == "train_end_epoch"):
            self.get_measures(batch)

        self.step = old_step
