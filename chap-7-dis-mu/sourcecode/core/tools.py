import inspect


def call_fun(fun, kwargs):

    kwargs = dict(kwargs)

    fun_param = list(inspect.signature(fun).parameters.keys())
    for key in list(kwargs.keys()):
        if(key not in fun_param):
            del kwargs[key]

    return fun(**kwargs)


def not_kwargs_fun(fun, kwargs):

    not_kwargs = dict(kwargs)

    fun_param = list(inspect.signature(fun).parameters.keys())
    for key in list(kwargs.keys()):
        if(key in fun_param):
            del not_kwargs[key]

    return not_kwargs

