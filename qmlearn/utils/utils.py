import numpy as np
import hashlib

def dict_update(d, u):
    import collections.abc
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def copy_array(arr, out = None, add = False, index = None):
    if index is None : index = slice(None)
    if out is None :
        out = arr.copy()
    elif add :
        out[index] += arr
    else :
        out[index] = arr
    return out

def get_hash(x):
    value = hashlib.md5(np.array(sorted(x))).hexdigest()
    return value

def matrix_deviation(mat1, mat2):
    diff = np.abs(mat1 - mat2)
    errorsum = np.sum(diff)
    return errorsum
