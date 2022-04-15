import numpy as np
import itertools as it
from ase.build.rotate import rotation_matrix_from_points
from ase.geometry import get_distances
from sklearn.decomposition import PCA
from rmsd.calculate_rmsd import (
    kabsch,
    quaternion_rotate,
    reorder_brute,
    reorder_distance,
    reorder_hungarian,
    reorder_inertia_hungarian,
    )

from qmlearn.utils.utils import matrix_deviation

class Engine(object):
    def __init__(self, mol = None, method = 'rks', basis = '6-31g', xc = None, **kwargs):
        self.options = locals()
        self.options.update(kwargs)
        #-----------------------------------------------------------------------
        self._vext = None
        self._kop = None
        self._gamma = None
        self._ovlp = None
        self._eri = None
        self.init()

    def init(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        self.orb = None
        self.gamma = None

    @property
    def vext(self):
        pass

    @property
    def kop(self):
        pass

    @property
    def ovlp(self):
        pass

    @property
    def ncharge0(self):
        pass

    def calc_gamma(self, orb=None, occs=None):
        pass

    def calc_ncharge(self, gamma, ovlp = None):
        if ovlp is None : ovlp = self.ovlp
        ncharge = np.einsum('ji,ij->', gamma, ovlp)
        return ncharge

    def calc_ke(self, gamma, kop = None):
        if kop is None : kop = self.kop
        ke = np.einsum('ji,ij->', gamma, kop)
        return ke

    def calc_idempotency(self, gamma, ovlp=None, kind=1):
        # Only for nspin=1
        if ovlp is None : ovlp = self.ovlp
        if kind==1:
            g = gamma@ovlp@gamma/2
        if kind==2:
            g = (3*gamma@ovlp@gamma@ovlp@gamma - 2*gamma@ovlp@gamma)/8
        if kind==3:
            g = (4*gamma@ovlp@gamma@ovlp@gamma -gamma@ovlp@gamma@ovlp@gamma@ovlp@gamma)/8
        return matrix_deviation(gamma, g)

def atoms_rmsd(target, atoms, transform = True, **kwargs) :
    if transform :
        op_rotate, op_translate, op_indices = minimize_rmsd_operation(target, atoms, **kwargs)
        positions = np.dot(atoms.positions,op_rotate)+op_translate
        atoms = atoms[op_indices]
        atoms.set_positions(positions)
    rmsd = rmsd_coords(target.positions, atoms.positions)
    return rmsd, atoms

def rmsd_coords(target, pos, weights = None):
    diff = pos - target
    if weights is not None :
        weights = np.asarray(weights)
        if weights.ndim == 1 and len(weights) > 1 :
            weights = weights[:, None]
        diff *= weights
    rmsd = np.sqrt(np.sum(diff*diff)/len(diff))
    return rmsd

def atoms2bestplane(atoms, direction = None):
    pca = PCA()
    pos = pca.fit_transform(atoms.positions)
    atoms.set_positions(pos)
    if direction is not None :
        atoms = atoms2newdirection(atoms, a=(0,0,1), b=direction)
    return atoms

def get_atoms_axes(atoms):
    pca = PCA(n_components=3)
    pca.fit(atoms.positions)
    axes = pca.components_
    return axes

def atoms2newdirection(atoms, a=(0,0,1), b=(1,0,0)):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float) / np.linalg.norm(a)
    b = np.asarray(b, dtype=float) / np.linalg.norm(b)
    if np.allclose(a, b) : return atoms
    c = np.cross(a, b)
    deg= np.rad2deg(np.arccos(np.clip(np.dot(a,b),-1.0,1.0)))
    atoms.rotate(deg,c)
    return atoms

def minimize_rmsd_operation(target, atoms, stereo = True, rotate_method = 'kabsch',
        reorder_method = 'hungarian', use_reflection = True, alpha = 0.2):
    pos_t = target.get_positions()
    c_t = np.mean(pos_t, axis=0)
    pos_t = pos_t - c_t

    pos = atoms.get_positions()
    c = np.mean(pos, axis=0)
    pos = pos - c
    #-----------------------------------------------------------------------
    atoms1 = target.copy()
    atoms1.positions[:] = pos_t
    atoms2 = atoms.copy()
    axes1 = np.abs(get_atoms_axes(atoms1))
    #-----------------------------------------------------------------------
    if use_reflection :
        srot=np.zeros((48,3,3))
        mr = np.array(list(it.product([1,-1], repeat=3)))
        i=0
        for swap in it.permutations(range(3)):
            for ijk in mr:
                srot[i][tuple([(0,1,2),swap])]= ijk
                i+=1
    else :
        srot=[np.eye(3)]
    #-----------------------------------------------------------------------
    rmsd_final_min = np.inf
    for rot in srot :
        if stereo and np.linalg.det(rot) < 0.0 : continue
        atoms2.set_positions(np.dot(pos, rot))
        atoms2.set_chemical_symbols(atoms.get_chemical_symbols())
        #
        indices = reorder_atoms_indices(atoms1, atoms2, reorder_method=reorder_method)
        atoms2 = atoms2[indices]
        rotate = get_match_rotate(atoms1, atoms2, rotate_method = rotate_method)
        atoms2.positions[:] = np.dot(atoms2.positions[:], rotate)
        rmsd = rmsd_coords(atoms1.positions, atoms2.positions)
        axes2 = np.abs(get_atoms_axes(atoms2))
        rmsd += rmsd_coords(axes1, axes2)*alpha
        if rmsd < rmsd_final_min :
            rmsd_final_min = rmsd
            rmsd_final_rotate = rotate
            rmsd_final_reflection = rot
            rmsd_final_indices = indices
    rotate = np.dot(rmsd_final_reflection, rmsd_final_rotate)
    translate = c_t - np.dot(c, rotate)
    # print('rmsd_final_min', rmsd_final_min)
    # positions = np.dot(atoms.positions, rotate) + translate
    # rmsd = rmsd_coords(target.positions, positions[rmsd_final_indices])
    # print('rmsd', rmsd)
    return rotate, translate, rmsd_final_indices

def get_match_rotate(target, atoms, rotate_method = 'kabsch'):
    if rotate_method is None or rotate_method == 'none':
        rotate = np.eye(3)
    else :
        if not hasattr(rotate_method, '__call__'):
            if rotate_method == 'kabsch' :
                rotate_method = kabsch
            elif rotate_method == 'quaternion' :
                rotate_method = quaternion_rotate
                raise AttributeError(f"Sorry, not support '{rotate_method}'.")
        rotate = rotate_method(atoms.positions, target.positions)
    return rotate

def reorder_atoms_indices(target, atoms, reorder_method='hungarian'):
    if reorder_method is None or reorder_method == 'none':
        indices = slice(None)
    else :
        if not hasattr(reorder_method, '__call__'):
            if reorder_method == 'hungarian' :
                reorder_method = reorder_hungarian
            elif reorder_method == 'inertia-hungarian' :
                reorder_method = reorder_inertia_hungarian
            elif reorder_method == 'brute' :
                reorder_method = reorder_brute
            elif reorder_method == 'distance' :
                reorder_method = reorder_distance
            else :
                raise AttributeError(f"Sorry, not support '{reorder_method}'.")
        indices = reorder_method(np.asarray(target.get_chemical_symbols()),
                np.asarray(atoms.get_chemical_symbols()), target.positions, atoms.positions)
    return indices

def _reorder_atoms_v0(target, atoms):
    from scipy.optimize import linear_sum_assignment
    symbols1 = np.asarray(target.get_chemical_symbols())
    symbols2 = np.asarray(atoms.get_chemical_symbols())
    slist = np.unique(symbols1)
    indices = np.zeros_like(symbols1, dtype=int)
    for s in slist :
        i1 = symbols1 == s
        i2 = symbols2 == s
        j2 = np.where(i2)[0]
        distances = get_distances(target[i1].positions, atoms[i2].positions)[1]
        _, inds = linear_sum_assignment(distances)
        indices[i1] = j2[inds]
    return atoms[indices]

def _minimize_rmsd_operation_v0(target, atoms):
    pos_t = target.get_positions()
    c_t = np.mean(pos_t, axis=0)
    pos_t = pos_t - c_t

    pos = atoms.get_positions()
    c = np.mean(pos, axis=0)
    pos = pos - c

    rotate = rotation_matrix_from_points(pos.T, pos_t.T).T
    translate = c_t - np.dot(c, rotate)
    index = np.arange(len(pos))
    return(rotate, translate, index)
