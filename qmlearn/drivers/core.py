import numpy as np
from ase.build.rotate import minimize_rotation_and_translation, rotation_matrix_from_points
from sklearn.decomposition import PCA

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

def atoms_rmsd(refatoms, atoms, keep = False, transform = True):
    if transform :
        if keep : atoms = atoms.copy()
        minimize_rotation_and_translation(refatoms, atoms)
    diff = atoms.positions - refatoms.positions
    rmsd = np.sqrt(np.sum(diff*diff)/len(diff))
    return rmsd, atoms

def atoms2bestplane(atoms, direction = None):
    pca = PCA()
    pos = pca.fit_transform(atoms.positions)
    atoms.set_positions(pos)
    if direction is not None :
        atoms = atoms2newdirection(atoms, a=(0,0,1), b=direction)
    return atoms

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

def minimize_rmsd_operation(target, atoms):
    pos_t = target.get_positions()
    c_t = np.mean(pos_t, axis=0)
    pos_t = pos_t - c_t

    pos = atoms.get_positions()
    c = np.mean(pos, axis=0)
    pos = pos - c

    rotate = rotation_matrix_from_points(pos.T, pos_t.T).T
    translate = c_t - np.dot(c, rotate)
    return(rotate, translate)
