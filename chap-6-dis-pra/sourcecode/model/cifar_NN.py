import copy
import math
import torch
# NOTE: Inspired from https://keras.io/examples/cifar10_resnet/


class ResnetLayer(torch.nn.Module):
    def __init__(
        self, device, kwargs,
        new_channel_size,
        old_channel_size,
        kernel_size=3,
        stride=1,
        padding=1,
        activation=True,
        batch_norm=True,
    ):
        super().__init__()
        self.__device = device

        # Saving parameters
        self.__var = 0.0001
        if("var" in kwargs):
            self.__var = float(kwargs["var"])
        self.__var = torch.tensor(self.__var)

        self.__stride = stride
        self.__padding = padding

        # Creating Batch Norm layer
        self._batch_norm = None
        if(batch_norm):
            self._batch_norm = torch.nn.BatchNorm2d(
                new_channel_size, affine=False, track_running_stats=False)

        # Creating activation layer
        self._activation = None
        if(activation):
            self._activation = torch.nn.ReLU()

        # Creating the weights
        self.__mean_linear = torch.nn.Parameter(torch.empty(
            new_channel_size, old_channel_size, kernel_size, kernel_size))
        self.__mean_bias = torch.nn.Parameter(
            torch.empty(new_channel_size))

        # Initializing weights with He initializer
        torch.nn.init.kaiming_normal_(self.__mean_linear)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.__mean_linear)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(
            self.__mean_bias, -bound, bound)

        # Copying the "prior" weights to have "posterior" weights
        self.__prior_mean_linear = copy.deepcopy(self.__mean_linear)
        self.__prior_mean_bias = copy.deepcopy(self.__mean_bias)

        self.__prior_mean_linear.requires_grad = False
        self.__prior_mean_bias.requires_grad = False

        self.prior_param_dict = {
            "linear": self.__prior_mean_linear,
            "bias": self.__prior_mean_bias,
        }
        self.post_param_dict = {
            "linear": self.__mean_linear,
            "bias": self.__mean_bias,
        }

        self.random_linear = None
        self.random_bias = None

    def forward(self, x, keep, div_renyi, div_rivasplata, batch):

        if(not(keep) or self.random_linear is None):
            # Inializing the random weights vector
            self.random_linear = torch.randn(
                self.__mean_linear.shape,
                device=self.__device
            )*torch.sqrt(self.__var)
        if(not(keep) or self.random_bias is None):
            self.random_bias = torch.randn(
                self.__mean_bias.shape,
                device=self.__device
            )*torch.sqrt(self.__var)

        # Forwarding in convolution and activation
        x = torch.nn.functional.conv2d(
            x, self.__mean_linear+self.random_linear,
            bias=self.__mean_bias+self.random_bias,
            stride=self.__stride,
            padding=self.__padding
        )

        # Forwarding in the batch norm and activation layer
        if(self._batch_norm is not None):
            x = self._batch_norm(x)
        if(self._activation is not None):
            x = self._activation(x)

        # Computing the divergence
        # ------------------------------------------------------------------- #
        div_renyi += torch.sum(
            (self.__prior_mean_linear
             - self.__mean_linear)**2)
        div_renyi += torch.sum(
            (self.__prior_mean_bias
             - self.__mean_bias)**2)

        div_rivasplata += torch.sum(
            (self.__mean_linear+self.random_linear
             - self.__prior_mean_linear)**2)
        div_rivasplata += torch.sum(
            (self.__mean_bias+self.random_bias
             - self.__prior_mean_bias)**2)
        div_rivasplata -= (
            torch.sum(self.random_linear**2)+torch.sum(self.random_bias**2))
        # ------------------------------------------------------------------- #

        return x, div_renyi, div_rivasplata


class Model(torch.nn.Module):

    def __update_param_dict(self):
        self.prior_param_dict["module_linear_list"].append(
            self.__module_list[
                len(self.__module_list)-1].prior_param_dict["linear"])
        self.prior_param_dict["module_bias_list"].append(
            self.__module_list[
                len(self.__module_list)-1].prior_param_dict["bias"])

        self.post_param_dict["module_linear_list"].append(
            self.__module_list[
                len(self.__module_list)-1].post_param_dict["linear"])
        self.post_param_dict["module_bias_list"].append(
            self.__module_list[
                len(self.__module_list)-1].post_param_dict["bias"])

    def __init__(self, device, **kwargs):
        super().__init__()

        self.prior_param_dict = {
            "module_linear_list": [],
            "module_bias_list": []
        }
        self.post_param_dict = {
            "module_linear_list": [],
            "module_bias_list": []
        }

        self.__device = device

        # We fix the depth of the network
        self.__depth = 20
        if("depth" in kwargs):
            self.__depth = int(kwargs["depth"])

        # We fix the number of classes
        self.__classes = 10
        self.__var = 0.0001
        if("var" in kwargs):
            self.__var = float(kwargs["var"])
        self.__var = torch.tensor(self.__var)

        # Checking the size of the network
        if (self.__depth - 2) % 6 != 0:
            raise Exception('Depth should be 6n+2 (e.g, 20, 32, 44)')

        # Start model definition.
        old_channel_size = 3
        new_channel_size = 32
        block_size = int((self.__depth - 2) / 6)
        stack_size = 3

        # Computing the shape after the laster layer
        self.__hidden_size = int(
            2**(math.log(new_channel_size, 2)-1+stack_size))

        # Creating the module list
        self.__module_list = torch.nn.ModuleList()
        # Inserting module in the list
        self.__module_list.append(
            ResnetLayer(device, kwargs, new_channel_size, old_channel_size)
        )

        self.__update_param_dict()

        old_channel_size = new_channel_size

        for stack in range(stack_size):
            for block in range(block_size):

                stride = 1
                # Downsampling when first layer for each stack (except 1)
                if(stack > 0 and block == 0):
                    stride = 2

                # Inserting module in the list
                self.__module_list.append(
                    ResnetLayer(
                        device, kwargs,
                        new_channel_size, old_channel_size, stride=stride)
                )

                self.__update_param_dict()

                self.__module_list.append(
                    ResnetLayer(
                        device, kwargs,
                        new_channel_size, new_channel_size, activation=None)
                )

                self.__update_param_dict()

                # Downsampling when first layer for each stack (except 1)
                # Linear projection to match dim
                if(stack > 0 and block == 0):
                    # Inserting module in the list
                    self.__module_list.append(
                        ResnetLayer(
                            device, kwargs,
                            new_channel_size, old_channel_size,
                            kernel_size=1, padding=0, stride=stride,
                            activation=None, batch_norm=None)
                    )
                    self.__update_param_dict()

                old_channel_size = new_channel_size
            new_channel_size *= 2

        # Creating weights for linear module
        self.__mean_linear = torch.nn.Parameter(torch.empty(
            self.__classes, self.__hidden_size))
        self.__mean_bias = torch.nn.Parameter(
            torch.empty(self.__classes))

        # Initializing weights with He initializer
        torch.nn.init.kaiming_normal_(self.__mean_linear)
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                self.__mean_linear)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(
            self.__mean_bias, -bound, bound)

        # Copying the "prior" weights to have "posterior" weights
        self.__prior_mean_linear = copy.deepcopy(self.__mean_linear)
        self.__prior_mean_bias = copy.deepcopy(self.__mean_bias)

        self.__prior_mean_linear.requires_grad = False
        self.__prior_mean_bias.requires_grad = False

        self.prior_param_dict["linear"] = self.__prior_mean_linear
        self.prior_param_dict["bias"] = self.__prior_mean_bias
        self.post_param_dict["linear"] = self.__mean_linear
        self.post_param_dict["bias"] = self.__mean_bias

        self.random_linear = None
        self.random_bias = None

    def forward(self, batch):
        x = batch["x"]

        keep = False
        if("keep" in batch):
            keep = batch["keep"]

        # NOTE: We consider that the input is (batch_size, 3, x, x),
        # where x is the height/width of the image
        x = torch.reshape(
            x, (x.shape[0], 3,
                int(math.sqrt(x.shape[1]/3)), int(math.sqrt(x.shape[1]/3))))

        # Computing the number of blocks
        block_size = int((self.__depth - 2) / 6)

        # Initializing the divergence
        div_renyi = 0.0
        div_rivasplata = 0.0

        x_out, div_renyi, div_rivasplata = self.__module_list[0](
            x, keep, div_renyi, div_rivasplata, batch)
        i = 1
        for stack in range(3):
            for block in range(block_size):

                x_res, div_renyi, div_rivasplata = self.__module_list[i](
                    x_out, keep, div_renyi, div_rivasplata, batch)
                x_res, div_renyi, div_rivasplata = self.__module_list[i+1](
                    x_res, keep, div_renyi, div_rivasplata, batch)
                i += 2

                if (stack > 0 and block == 0):
                    x_out, div_renyi, div_rivasplata = self.__module_list[i](
                        x_out, keep, div_renyi, div_rivasplata, batch)
                    i += 1

                x_out = x_res + x_out
                x_out = torch.nn.functional.relu_(x_out)

        # Forwarding in the average pooling
        x_out = torch.nn.functional.avg_pool2d(x_out, 8)
        x_out = torch.squeeze(x_out)

        if(not(keep) or self.random_linear is None):
            self.random_linear = torch.randn(
                self.__mean_linear.shape,
                device=self.__device
            )*torch.sqrt(self.__var)
        if(not(keep) or self.random_bias is None):
            self.random_bias = torch.randn(
                self.__mean_bias.shape,
                device=self.__device
            )*torch.sqrt(self.__var)

        # Forwarding in the linear layer
        x = torch.nn.functional.linear(
            x_out, self.__mean_linear+self.random_linear,
            bias=self.__mean_bias+self.random_bias)

        self.out = torch.nn.functional.softmax(x, dim=1)
        self.pred = torch.max(x, dim=1)[1].unsqueeze(1)

        # Computing the divergence
        # ------------------------------------------------------------------- #
        div_renyi += torch.sum(
            (self.__prior_mean_linear
             - self.__mean_linear)**2)
        div_renyi += torch.sum(
            (self.__prior_mean_bias
             - self.__mean_bias)**2)

        div_rivasplata += torch.sum(
            (self.__mean_linear+self.random_linear
             - self.__prior_mean_linear)**2)
        div_rivasplata += torch.sum(
            (self.__mean_bias+self.random_bias
             - self.__prior_mean_bias)**2)
        div_rivasplata -= (
            torch.sum(self.random_linear**2)+torch.sum(self.random_bias**2))
        # ------------------------------------------------------------------- #

        # and dividing by the variance ;)
        self.div_renyi = (div_renyi/self.__var)
        self.div_kl = 0.5*self.div_renyi
        self.div_rivasplata = (0.5/self.__var)*div_rivasplata
