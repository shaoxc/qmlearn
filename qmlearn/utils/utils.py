import numpy as np
import hashlib
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

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

def array2blocks(arr, sections):
    blocks={}
    ix=0
    ns=len(sections)
    for i in range(ns):
        iy=0
        for j in range(ns):
            indices = np.s_[ix:ix+sections[i],iy:iy+sections[j]]
            blocks[(i,j)] = arr[indices]
            iy=iy+sections[j]
        ix=ix+sections[i]
    return blocks

def blocks2array(blocks, indices):
    new_blocks=[]
    for i in indices:
        la=[]
        for j in indices:
            la.append(blocks[(i,j)])
        new_blocks.append(la)
    arr = np.block(new_blocks)
    return arr

def wedge(a, b=None):
    if b is None: b = a
    ab1 = np.einsum('pq,rs->pqrs',a,b) + np.einsum('pq,rs->rspq',a,b)
    ab2 = np.einsum('pq,rs->prsq',a,b) + np.einsum('pq,rs->sqpr',a,b)
    return (ab1-ab2)*0.25

def unitary_decompose(amat, a=None, trace=None, r=None, identity=None):
    if a is None: a = np.trace(amat)
    if trace is None: trace = np.einsum('iijj->', amat)
    if r is None: r = amat.shape[0]
    if identity is None: identity = np.eye(r)
    A0 = 2.0 * trace / (r*(r-1)) * wedge(identity)
    A1_0 = 4.0 / (r-2) * wedge(a, identity)
    A1 = A1_0 - 2*(r-1)/(r-2) * A0
    A2 = amat - A1 - A0
    return A0, A1, A2
