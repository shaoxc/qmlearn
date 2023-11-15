import inspect
from importlib import import_module
import logging

SKLEARN_PROPS={
        'LinearRegression' : {
            'coef_' : 'array',
            'rank_' : 'int',
            'singular_' : 'int',
            'intercept_': 'array',
            'n_features_in_' : 'int',
            'feature_names_in_': 'array',
            },
        'KernelRidge' : {
            'dual_coef_' : 'array',
            'X_fit_' : 'array',
            },
        }


def model_to_dict(model, props=None):
    module = inspect.getmodule(model)
    p = import_module(module.__name__.partition('.')[0])
    version = getattr(p, '__version__', None)
    name = model.__class__.__name__
    values = {
        '__version__': p.__name__+','+version,
        '__module__': module.__name__,
        '__name__': name,
        'params': model.get_params()
    }
    if props is None:
        if name not in SKLEARN_PROPS :
            raise AttributeError(f'The "{name}" is not supported yet.')
        props = SKLEARN_PROPS.get(values['__name__'], [])
    for k in props:
        values[k] = getattr(model, k, None)
    return values

def dict_to_model(values, props=None):
    p, version = values['__version__'].split(',')
    p = import_module(p)
    __version__ = getattr(p, '__version__', None)
    if __version__ != version:
        logging.warning(f'The versions of saved and installed of {p.__name__} are different: {version}--> {__version__}.')

    module = import_module(values['__module__'])
    cls = getattr(module, values['__name__'])
    model = cls(**values['params'])
    if props is None:
        props = [k for k in values if not k.startswith('__')]

    for k in props :
        v = values.get(k, None)
        if v is not None :
            setattr(model, k, v)

    return model
