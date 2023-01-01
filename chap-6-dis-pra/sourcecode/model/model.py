import re
import sys
import glob
import os
import importlib


class MetaModel(type):

    def __get_module_dict(cls):
        # Getting the current path, the file path and the module directory path
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

        module_dict = {}
        for module in sys.modules:
            if(re.match(r"^model[.].+", module)):
                module_class = sys.modules[module].Model
                module = module.replace("model.", "")
                module_dict[module] = module_class
        return module_dict

    def __call__(cls, *args, **kwargs):
        # Initializing the base classes
        bases = (cls, )

        # Getting the name of the module
        module_name = args[0]

        # Getting the module dictionnary
        module_dict = cls.__get_module_dict()

        # Checking that the module exists
        if(module_name not in module_dict):
            raise Exception(module_name+" doesn't exist")

        # Adding the new module in the base classes
        bases += (module_dict[module_name], )

        # Creating the new object with the good base classes
        new_cls = type(cls.__name__, bases, {})
        return super(MetaModel, new_cls).__call__(*args, **kwargs)


class Model(metaclass=MetaModel):

    def __init__(self, name, device, **kwargs):
        super().__init__(device, **kwargs)
