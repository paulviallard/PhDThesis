import copy
import math
import torch


class Model(torch.nn.Module):

    def __init__(self, device, **kwargs):
        super().__init__()

        self.__device = device

        # Initializing variance
        self.__var = 0.0001
        if("var" in kwargs):
            self.__var = float(kwargs["var"])
        self.__var = torch.tensor(self.__var)

        self.alpha = 2.0

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

        # Copying the "prior" weights to have "posterior" weights
        self.__prior_mean_linear = copy.deepcopy(self.__mean_linear)
        self.__prior_mean_bias = copy.deepcopy(self.__mean_bias)
        self.__prior_mean_linear.requires_grad = False
        self.__prior_mean_bias.requires_grad = False

        # For the learner
        self.prior_param_dict = {
            "linear": self.__prior_mean_linear,
            "bias": self.__prior_mean_bias,
        }
        self.post_param_dict = {
            "linear": self.__mean_linear,
            "bias": self.__mean_bias,
        }

        self.random_linear = [None for _ in range(len(self.__mean_linear))]
        self.random_bias = [None for _ in range(len(self.__mean_linear))]

    def forward(self, batch):
        x = batch["x"]

        x = torch.reshape(x, (
            x.shape[0], 1,
            int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))))

        keep = False
        if("keep" in batch):
            keep = batch["keep"]

        div_renyi = 0.0
        div_rivasplata = 0.0

        # Forwarding in the layers
        for i in range(len(self.__mean_linear)):

            if(not(keep) or self.random_linear[i] is None):
                # Computing the random vector
                self.random_linear[i] = torch.randn(
                    self.__mean_linear[i].shape,
                    device=self.__device
                )*torch.sqrt(self.__var)
            if(not(keep) or self.random_bias[i] is None):
                self.random_bias[i] = torch.randn(
                    self.__mean_bias[i].shape,
                    device=self.__device
                )*torch.sqrt(self.__var)

            # Forwarding in convolution and activation
            x = torch.nn.functional.conv2d(
                x, (self.__mean_linear[i])+self.random_linear[i],
                bias=(self.__mean_bias[i])+self.random_bias[i],
                stride=self.__param_list[i][0],
                padding=self.__param_list[i][1]
            )
            x = torch.nn.functional.leaky_relu(x)

            # --------------------------------------------------------------- #
            div_renyi += torch.sum(
                (self.__prior_mean_linear[i]
                 - self.__mean_linear[i])**2)
            div_renyi += torch.sum(
                (self.__prior_mean_bias[i]
                 - self.__mean_bias[i])**2)

            div_rivasplata += torch.sum(
                (self.__mean_linear[i]+self.random_linear[i]
                 - self.__prior_mean_linear[i])**2)
            div_rivasplata += torch.sum(
                (self.__mean_bias[i]+self.random_bias[i]
                 - self.__prior_mean_bias[i])**2)
            div_rivasplata -= (
                torch.sum(self.random_linear[i]**2)
                + torch.sum(self.random_bias[i]**2))
            # --------------------------------------------------------------- #

        # Forwarding average pooling
        x = torch.nn.functional.avg_pool2d(x, 8)
        x = torch.squeeze(x)

        self.out = torch.nn.functional.softmax(x, dim=1)
        self.pred = torch.max(x, dim=1)[1].unsqueeze(1)

        self.div_renyi = (self.alpha/2.0)*(div_renyi/self.__var)
        self.div_kl = 0.5*self.div_renyi
        self.div_rivasplata = (0.5/self.__var)*div_rivasplata
