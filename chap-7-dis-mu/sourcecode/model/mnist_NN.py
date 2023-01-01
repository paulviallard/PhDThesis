import math
import torch


class Model(torch.nn.Module):

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.step = "train"

        # NOTE: Assuming 10 classes
        # Iniatializing the parameters (mean and variance of gaussian)
        self.__mean_linear = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(96, 1, 5, 5)),
            torch.nn.Parameter(torch.empty(96, 96, 5, 5)),
            torch.nn.Parameter(torch.empty(96, 96, 5, 5)),
            torch.nn.Parameter(torch.empty(10, 96, 5, 5)),
        ])
        self.__mean_bias = torch.nn.ParameterList([
            torch.nn.Parameter(torch.empty(96)),
            torch.nn.Parameter(torch.empty(96)),
            torch.nn.Parameter(torch.empty(96)),
            torch.nn.Parameter(torch.empty(10)),
        ])

        # Initializing stride and padding
        self.__param_list = [[1, 1], [2, 1], [1, 1], [1, 1]]

        # Initializing weights with Xavier initializer
        for i in range(len(self.__mean_linear)):
            torch.nn.init.xavier_normal_(self.__mean_linear[i])
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.__mean_linear[i])
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(
                self.__mean_bias[i], -bound, bound)

        self.random_linear = [None for _ in range(len(self.__mean_linear))]
        self.random_bias = [None for _ in range(len(self.__mean_linear))]

        # ------------------------------------------------------------------- #

        self.step = "measure"
        self._init_param = []
        for param in self.parameters():
            param = param.clone().detach()
            self._init_param.append(param)
        self.step = None

    def forward(self, batch):
        x = batch["x"]
        old_step = self.step
        self.step = batch["step"]

        # Forwarding in the layers
        for i in range(len(self.__mean_linear)):

            # Forwarding in convolution and activation
            x = torch.nn.functional.conv2d(
                x, (self.__mean_linear[i]),
                bias=(self.__mean_bias[i]),
                stride=self.__param_list[i][0],
                padding=self.__param_list[i][1]
            )
            x = torch.nn.functional.leaky_relu(x)

        # Forwarding average pooling
        x = torch.nn.functional.avg_pool2d(x, 8)
        x = torch.squeeze(torch.squeeze(x, dim=2), dim=2)

        self.raw_out = x
        self.out = torch.nn.functional.softmax(x, dim=1)
        self.pred = torch.max(x, dim=1)[1].unsqueeze(1)
        if(self.step == "train" or self.step == "train_end_epoch"):
            self.get_measures(batch)

        self.step = old_step
