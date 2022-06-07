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
    r"""Abstract Base class for the External calculator.
    Until now we have implemented: PySCF and Psi4.

    Attributes
    ----------

    vext : ndarray
        External Potential.
    gamma : ndarray
        1-body reduced density matrix (1-RDM).
    gammat : ndarray
        2-body reduced density matrix (2-RDM).
    etotal : float
        Total electronic energy
    forces : ndarray
        Atomic forces
    ovlp : ndarray
        Overlap Matrix.
    kop : ndarray
        Kinetic Energy Operator.
    eri : ndarray
        2-electron integrals: 8-fold or 4-fold ERIs or complex integral array with N^4 elements
        (N is the number of orbitals).
    orb : ndarray
        Molecular orbital coefficients
    """
    def __init__(self, mol = None, method = 'rks', basis = '6-31g', xc = None, **kwargs):
        self.options = locals()
        self.options.update(kwargs)
        #-----------------------------------------------------------------------
        self._vext = None
        self._gamma = None
        self._gammat= None
        self._etotal = None
        self._forces = None
        #
        self._kop = None
        self._ovlp = None
        self._eri = None
        self._orb = None

    def init(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        pass

    @property
    def gamma(self):
        r""" 1-body reduced density matrix (1-RDM). """
        pass

    @property
    def gammat(self):
        r""" 2-body reduced density matrix (2-RDM). """
        pass

    @property
    def etotal(self):
        r""" Total Energy. """
        pass

    @property
    def forces(self):
        r""" Atomic Forces.  """
        pass

    @property
    def vext(self):
        r""" External Potential. """
        pass

    @property
    def kop(self):
        r""" Kinetic Energy Operator. """
        pass

    @property
    def ovlp(self):
        r"""Overlap Matrix."""
        pass

    @property
    def ncharge0(self):
        r""" Calculated number of electrons. """
        pass

    def calc_gamma(self, orb=None, occs=None):
        r"""Calculate the 1-body reduced density matrix (1-RDM).

        Parameters
        ----------
        orb : ndarray
           Orbital Coefficients. Each column is one orbital
        occs : ndarray
            Occupancy

        Returns
        -------
        1-RDM : ndarray
        """
        pass

    def calc_ncharge(self, gamma, ovlp = None):
        r""" Get calculated total number of electrons

        Parameters
        ----------
        gamma : ndarray
            1-RDM 
        ovlp : ndarray 
            Overlap Matrix

        Returns
        -------
        ncharge : int
            Calcualted number of electrons

        """
        if ovlp is None : ovlp = self.ovlp
        ncharge = np.einsum('ji,ij->', gamma, ovlp)
        return ncharge

    def calc_ke(self, gamma, kop = None):
        r""" Get Total kinetic energy

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        kop : ndarray 
            Kinetic energy operator

        Returns
        -------
        ke : float
            Total kinetic energy

        """
        if kop is None : kop = self.kop
        ke = np.einsum('ji,ij->', gamma, kop)
        return ke

    def calc_idempotency(self, gamma, ovlp=None, kind=1):
        r""" Check idempotency of gamma  

        .. math::
           \gamma S \gamma &= 2 \gamma 

        Parameters
        ----------
        gamma : ndarray
            1-RDM
        ovlp : ndarray
            Overlap matrix

        Attributes
        ----------
        kind :  int

            | 1 : Level 1 
            | 2 : Level 2
            | 3 : Level 3

        Returns
        -------
        errorsum : float
            Sum over the Absolute difference among matrix 1 and 2.

        """
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
    r""" Function to return RMSD : Root mean square deviation between atoms and target:transform atom object. And the target atom coordinates.

    Parameters
    ----------
    target : :obj: ASE atoms object
        Reference atoms.
    atoms : :obj: ASE atoms object
        Atoms to be transformed (Rotate y/o translate).

    Attributes
    ----------
    keep : bool
        If True keep a copy of atoms.
    transform : bool
        If True minimize rotation and translation between refatoms and atoms. 
        (See ASE documentation: Minimize RMSD between atoms and target.)

    Returns
    -------
    rmsd : float
        RMSD betwwen reference and transform atoms.
    atoms : :obj: ASE atoms object
        Transformed target molecule.
    """
    if transform :
        op_rotate, op_translate, op_indices = minimize_rmsd_operation(target, atoms, **kwargs)
        positions = np.dot(atoms.positions,op_rotate)+op_translate
        atoms = atoms[op_indices]
        atoms.set_positions(positions[op_indices])
    rmsd = rmsd_coords(target.positions, atoms.positions)
    return rmsd, atoms

def rmsd_coords(target, pos, **kwargs):
    r""" Function to return RMSD : Root mean square deviation between pos and target atomic positions.

    Parameters
    ----------
    pos : :obj: ASE atoms object
        Reference atoms.
    target : :obj: ASE atoms object
        Atoms to be transformed (Rotate y/o translate).

    Returns
    -------
    rmsd : float
        RMSD between pos and target atomic positions  """
    return diff_coords(target, pos, diff_method='rmsd', **kwargs)

def diff_coords(target, pos = None, weights = None, diff_method = 'rmsd'):
    r""" Function to return mean base on method error/deviation between pos and target atomic positions.

    Parameters
    ----------
    pos : :obj: ASE atoms object
        Reference atoms.
    target : :obj: ASE atoms object
        Atoms to be transformed (Rotate y/o translate).

    Attributes
    ----------
    diff_method : str

        | RMSD(root-mean-square deviation) : 'rsmd'  
        | RMSE(root-mean-square error) : 'rmse'  
        | MSD(mean-square deviation) : 'msd'  
        | MSE(mean-square error) : 'mse' 
        | MAE(mean absolute error) : 'mae'

    Returns
    -------
    rmsd : float
        Mean base on method error/deviation between pos and target atomic positions """
    if pos is None :
        diff = target
    else :
        diff = pos - target
    if weights is not None :
        weights = np.asarray(weights)
        if weights.ndim == 1 and len(weights) > 1 :
            weights = weights[:, None]
        diff *= weights
    if diff_method in ['msd', 'mse'] : # mean squared deviation (MSD) or mean squared error (MSE)
        rmsd = np.sum(diff*diff)/len(diff)
    elif diff_method in ['rmsd', 'rmse'] : # root-MSD or root-MSE
        rmsd = np.sqrt(np.sum(diff*diff)/len(diff))
    elif diff_method == 'mae' : # mean absolute error (MAE)
        rmsd = np.sum(np.abs(diff))/len(diff)
    return rmsd

def atoms2bestplane(atoms, direction = None):
    r""" Apply Principal Component Analysis (PCA) to atoms and re-oriente them.

    Parameters
    ----------
    atoms : :obj: ASE atoms object
       Atoms coordinates 

    Attributes
    ----------
    direction : ndarray
        1D-vector to re-orient the atoms positions

    Returns
    -------
    atoms : ndarray
        Reoriented atom positions 
    """
    pca = PCA()
    pos = pca.fit_transform(atoms.positions)
    atoms.set_positions(pos)
    if direction is not None :
        atoms = atoms2newdirection(atoms, a=(0,0,1), b=direction)
    return atoms

def get_atoms_axes(atoms):
    r""" Get PCA components of atomic positions.
            
    Parameters
    ----------
    atoms : :obj: ASE object

    Returns
    -------
    axes : ndarry
        Vector containing the PCA components of the atomic positions """
    pca = PCA(n_components=3)
    pca.fit(atoms.positions)
    axes = pca.components_
    return axes

def atoms2newdirection(atoms, a=(0,0,1), b=(1,0,0)):
    r""" Function to re-orientate the atom positions. 
    If Vector is None, the default rotation is around X-axis.

    Parameters
    ----------
    atoms : :obj: ASE atom object
    
    Returns
    -------
    atoms : ndarray
        Rotated atoms coordinates
    """
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
    r""" Function to create Rotation Matrix and Translation Vector
    of reference atoms with respect to initialize atoms.

    Parameters
    ----------
    target : :obj: ASE atoms object
        References atoms
    atoms : :obj: ASE atoms object
        Initialize atoms

    Returns
    -------
    rotate : ndarray
        Rotation Matrix
    translate : ndarray
        Translation Vector
    rmsd_final_indices : ndarry
        Reorderred atom indices 
    """
    # return _minimize_rmsd_operation_v0(target, atoms)
    pos_t = target.get_positions()
    c_t = np.mean(pos_t, axis=0)
    # c_t = target.get_center_of_mass()
    pos_t = pos_t - c_t

    pos = atoms.get_positions()
    c = np.mean(pos, axis=0)
    # c = atoms.get_center_of_mass()
    pos = pos - c
    #-----------------------------------------------------------------------
    atoms1 = target.copy()
    atoms1.positions[:] = pos_t
    atoms2 = atoms.copy()
    # axes1 = np.abs(get_atoms_axes(atoms1))
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
    for ia, rot in enumerate(srot):
        if stereo and np.linalg.det(rot) < 0.0 : continue
        atoms2.set_positions(np.dot(pos, rot))
        atoms2.set_chemical_symbols(atoms.get_chemical_symbols())
        #
        indices = reorder_atoms_indices(atoms1, atoms2, reorder_method=reorder_method)
        atoms2 = atoms2[indices]
        rotate = get_match_rotate(atoms1, atoms2, rotate_method = rotate_method)
        atoms2.positions[:] = np.dot(atoms2.positions[:], rotate)
        rmsd = diff_coords(atoms1.positions, atoms2.positions, diff_method = 'mae')
        # if rmsd < 0.3 :
            # print('r0', rmsd, ia, rmsd_final_min)
            # atoms2.write('try_' + str(ia) + '.xyz')
        # axes2 = np.abs(get_atoms_axes(atoms2))
        # rmsd += rmsd_coords(axes1, axes2)*alpha
        if rmsd < rmsd_final_min :
            rmsd_final_min = rmsd
            rmsd_final_rotate = rotate
            rmsd_final_reflection = rot
            rmsd_final_indices = indices
        # print('rmsd', rmsd, rmsd_final_min)
    rotate = np.dot(rmsd_final_reflection, rmsd_final_rotate)
    translate = c_t - np.dot(c, rotate)
    # print('rmsd_final_min', rmsd_final_min)
    # positions = np.dot(atoms.positions, rotate) + translate
    # rmsd = rmsd_coords(target.positions, positions[rmsd_final_indices])
    # print('rmsd', rmsd)
    return rotate, translate, rmsd_final_indices

def get_match_rotate(target, atoms, rotate_method = 'kabsch'):
    r""" Rotate Atoms with a customize method

    Parameters
    ----------
    target : :obj: ASE atoms object
        References atoms
    atoms : :obj: ASE atoms object
        Initialize atoms

    Attributes
    ----------
    rotate_metod: str
        
        | None : 'none'
        | Kabsch : 'kabsch'
        | Quaternion : 'quaternion'

    Returns
    -------
    rotate : ndarray
        Rotated atom positions """
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
    r""" Reorder atoms indices

    Parameters
    ----------
    target : :obj: ASE atoms object
        References atoms
    atoms : :obj: ASE atoms object
        Initialize atoms

    Returns
    -------
    indices : ndarray
        Reordered atom indices """
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
