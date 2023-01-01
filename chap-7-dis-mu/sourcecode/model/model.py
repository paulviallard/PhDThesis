import re
import sys
import glob
import os
import importlib
import torch
from core.augmentation import Augmentation
from torchvision import transforms
from core.module import Module
from learner.optimize_gd_learner import OptimizeGDLearner as learner


class MetaModel(type):

    def __get_model_dict(cls):
        # Getting the current path, the file path and the model directory path
        cwd_path = os.getcwd()
        file_path = os.path.dirname(__file__)

        os.chdir(file_path)
        import_module_list = glob.glob("*.py")
        import_module_list.remove("model.py")
        for import_module in import_module_list:
            import_module = import_module.replace(".py", "")
            import_module = "."+import_module
            importlib.import_module(import_module, package="model")

        # Setting back the old current directory
        os.chdir(cwd_path)

        model_dict = {}
        for model in sys.modules:
            if(re.match(r"^model[.].+", model)):
                model_class = sys.modules[model].Model
                model = model.replace("model.", "")
                model_dict[model] = model_class
        return model_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, )

        # Getting the name of the model
        model_name = args[0]

        # Getting the model dictionnary
        model_dict = cls.__get_model_dict()

        # Checking that the model exists
        if(model_name not in model_dict):
            raise Exception(model_name+" doesn't exist")

        # Adding the new model in the base classes
        bases += (model_dict[model_name], )

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaModel, new_cls).__call__(*args, **kwargs)


class Model(metaclass=MetaModel):

    def __init__(self, name, device, kwargs):

        self.name = name
        self.kwargs = kwargs
        self.zero_tensor = torch.tensor(0.0, device=device)
        super().__init__(device)

    ###########################################################################

    def get_param_list(self, init=False):

        param_list = list(self.parameters())
        if(init):
            param_list = self._init_param

        param_list_cat = []
        for param in param_list:
            param_list_cat.append(param.view(-1))
        param_list_cat = torch.cat(param_list_cat, dim=0)
        return param_list, param_list_cat

    ###########################################################################

    def get_path_norm(self, batch):

        batch = dict(batch)
        batch["step"] = "measure"

        x_shape = list(batch["x"].shape)
        x_shape[0] = 1
        old_x = batch["x"]
        batch["x"] = torch.ones(x_shape, device=self.device)

        old_raw_out = self.raw_out
        old_out = self.out
        old_pred = self.pred
        old_conv2d = torch.nn.functional.conv2d
        old_linear = torch.nn.functional.linear

        def path_linear(input, weight, bias=None):
            return old_linear(input, weight, bias)

        def path_conv2d(input, weight, bias=None,
                        stride=1, padding=0, dilation=1, groups=1):
            return old_conv2d(input, weight**2.0, bias**2.0,
                              stride, padding, dilation, groups)

        torch.nn.functional.conv2d = path_conv2d
        torch.nn.functional.linear = path_linear

        self(batch)
        norm = torch.sum(self.raw_out)

        torch.nn.functional.conv2d = old_conv2d
        torch.nn.functional.linear = old_linear
        batch["x"] = old_x
        self.raw_out = old_raw_out
        self.out = old_out
        self.pred = old_pred

        return norm

    def risk(self, batch):
        if(batch["step"] == "train"):
            new_out = learner(
                self, None, self.device, batch_size=64).output(batch["x"])
            risk = Module("BoundedCrossEntropy", self).fit
            risk = risk(new_out, batch["y"])
        else:
            new_pred = learner(
                self, None, self.device, batch_size=64).predict(batch["x"])
            risk = Module("ZeroOne", self).fit
            risk = risk(new_pred, batch["y"])
        return risk

    def risk_aug(self, batch):

        old_raw_out = self.raw_out
        old_out = self.out
        old_pred = self.pred
        old_x = batch["x"]
        old_step = batch["step"]
        batch["step"] = "measure"

        augmentation = Augmentation(
            [transforms.RandomAffine(degrees=20, translate=(0.1, 0.1))])
        batch["x"] = augmentation.fit(old_x)

        batch["step"] = old_step
        risk = self.risk(batch)

        self.raw_out = old_raw_out
        self.out = old_out
        self.pred = old_pred
        batch["x"] = old_x

        return risk

    ###########################################################################

    def get_measures(self, batch):
        # Based on https://github.com/nitarshan/robust-generalization-measures/
        # blob/master/data/generation/measures.py
        measures = {}

        if("step" in batch):
            old_step = self.step
            self.step = batch["step"]

        init_param, init_param_cat = self.get_param_list(init=True)
        param, param_cat = self.get_param_list()

        m = self.kwargs["measure"]

        #  Zero
        if(m in ["zero", "zero-aug"]):
            measures["zero"] = self.zero_tensor

        # ------------------------------------------------------------------- #

        #  Vector Norm Measures
        if(m in ["dist_l2", "dist_l2-aug"]):
            measures["dist_l2"] = (param_cat-init_param_cat).norm(p=2)

        # ------------------------------------------------------------------- #

        #  Frobenius Norm
        if(m in ["sum_fro", "sum_fro-aug", "dist_fro", "dist_fro-aug",
                 "param_norm", "param_norm-aug"]):
            fro_norm_list = param[0].norm("fro").unsqueeze(0)
            for i in range(1, len(param)):
                fro_norm_list = torch.cat(
                    (fro_norm_list, param[i].norm("fro").unsqueeze(0)))
            d = float(len(fro_norm_list))

            dist_fro_norm_list = (
                param[0]-init_param[0]).norm("fro").unsqueeze(0)
            for i in range(1, len(param)):
                dist_fro_norm_list = torch.cat(
                    (dist_fro_norm_list,
                     (param[i]-init_param[i]).norm("fro").unsqueeze(0)))

            prod_fro = (fro_norm_list**2.0).prod()

        if(m in ["sum_fro", "sum_fro-aug"]):
            measures["sum_fro"] = (d*prod_fro**(1/d))
        if(m in ["dist_fro", "dist_fro-aug"]):
            measures["dist_fro"] = (dist_fro_norm_list**2.0).sum()
        if(m in ["param_norm", "param_norm-aug"]):
            measures["param_norm"] = (fro_norm_list**2.0).sum()

        # ------------------------------------------------------------------- #

        #  Path Norm
        if(m in ["path_norm", "path_norm-aug"]):
            measures["path_norm"] = self.get_path_norm(batch)

        # ------------------------------------------------------------------- #

        # Data Augmentation
        if(m in ["zero-aug", "dist_l2-aug", "sum_fro-aug",
                 "dist_fro-aug", "path_norm-aug", "param_norm-aug"]):
            risk_aug = self.risk_aug(batch)
            risk = self.risk(batch)
            alpha = batch["alpha"]

        if(m in ["zero-aug"]):
            measures["zero-aug"] = (
                alpha*0.5*risk_aug-alpha*0.5*risk + measures["zero"])
        if(m in ["dist_l2-aug"]):
            measures["dist_l2-aug"] = (
                alpha*0.5*risk_aug-alpha*0.5*risk + measures["dist_l2"])
        if(m in ["sum_fro-aug"]):
            measures["sum_fro-aug"] = (
                alpha*0.5*risk_aug-alpha*0.5*risk + measures["sum_fro"])
        if(m in ["dist_fro-aug"]):
            measures["dist_fro-aug"] = (
                alpha*0.5*risk_aug-alpha*0.5*risk + measures["dist_fro"])
        if(m in ["path_norm-aug"]):
            measures["path_norm-aug"] = (
                alpha*0.5*risk_aug-alpha*0.5*risk + measures["path_norm"])
        if(m in ["param_norm-aug"]):
            measures["param_norm-aug"] = (
                alpha*0.5*risk_aug-alpha*0.5*risk + measures["param_norm"])

        # ------------------------------------------------------------------- #

        self.measures = measures
        self.step = old_step
